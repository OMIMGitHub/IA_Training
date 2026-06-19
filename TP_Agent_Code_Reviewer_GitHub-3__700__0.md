# TP — Agent Revieweur de Code avec GitHub
## De N8N à `create_agent` — Rendre visible ce que l'interface cache

**Formation Agentique IA · Mai 2026**  
**Durée estimée : 1h30 — Python 3.10+ · OpenRouter · GitHub API**

---

## Contexte

Dans le cours du matin, vous avez vu que N8N utilise sous le capot la même boucle ReAct que l'`AgentExecutor` de LangChain — mais en cache tout : les étapes intermédiaires, le raisonnement, les erreurs.

Ce TP suit un chemin en **trois actes** pour rendre tout cela visible, appliqué à un cas concret : un **agent de revue de code qui interagit avec GitHub en temps réel**.

```
Acte 1 — AgentExecutor + tools GitHub réels → observer la boucle ReAct
Acte 2 — Migration vers create_agent + suggestions inline
Acte 3 — Chaînage multi-agent → collision du silo mémoire
```

---

## Mise en place

### Stack requise

```bash
pip install langchain==1.2.0 langchain-core==1.2.5 \
  langchain-openai==1.1.6 langgraph==1.0.5 \
  langgraph-prebuilt==1.0.5 langchain-classic \
  python-dotenv requests
```

> **Pourquoi `langchain-classic` ?** Depuis LangChain 1.0, l'`AgentExecutor` et `create_tool_calling_agent` ont été extraits du package principal `langchain` et déplacés dans `langchain-classic`. L'Acte 1 (API legacy) en a besoin ; l'Acte 2 et l'Acte 3 (`create_agent`) ne l'utilisent pas. C'est déjà un premier signal de ce que ce TP veut montrer : la boucle ReAct n'a pas changé, mais l'API « historique » est désormais reléguée dans un package à part.

### Fichier `.env`

```env
OPENROUTER_API_KEY=sk-or-...
GITHUB_TOKEN=ghp_...
GITHUB_REPO=owner/repo-name        # ex: moncompte/tp-review-code
GITHUB_PR_NUMBER=1                  # numéro de la PR à reviewer
```

### Repo GitHub de test

Créez un repo public avec une PR ouverte contenant ce fichier `calculator.py` :

```python
# calculator.py — fichier à reviewer (intentionnellement défectueux)
def diviser(x, y):
    try:
        resultat = x / y
        print(resultat)
        return resultat
    except:
        pass

def additionner(a,b):
    r=a+b
    return r

def traiter_liste(items):
    for i in range(len(items)):
        print(items[i])
```

> **Astuce** : créez une branche `feature/calculator`, committez ce fichier, puis ouvrez une PR vers `main`. Notez le numéro de la PR dans votre `.env`.

---

## Acte 1 — Reconnaissance

**Objectif :** faire tourner l'`AgentExecutor` avec des tools GitHub réels et observer ce que N8N vous cachait.

### Code de départ

Créez un fichier `acte1.py` :

```python
import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# ── Configuration GitHub ───────────────────────────────────────────────────────
GITHUB_TOKEN  = os.environ["GITHUB_TOKEN"]
GITHUB_REPO   = os.environ["GITHUB_REPO"]
PR_NUMBER     = int(os.environ["GITHUB_PR_NUMBER"])
HEADERS       = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE_URL      = f"https://api.github.com/repos/{GITHUB_REPO}"


# ── Tools GitHub réels ─────────────────────────────────────────────────────────
@tool
def get_pr_files(pr_number: int) -> str:
    """Récupère la liste des fichiers modifiés dans une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur GitHub {resp.status_code} : {resp.text}"
    files = resp.json()
    result = []
    for f in files:
        result.append(
            f"Fichier : {f['filename']}\n"
            f"Statut  : {f['status']}\n"
            f"Lignes+ : {f['additions']} | Lignes- : {f['deletions']}\n"
            f"Patch   :\n{f.get('patch', '(pas de patch)')}\n"
        )
    return "\n---\n".join(result)


@tool
def post_review_comment(pr_number: int, body: str) -> str:
    """Poste un commentaire général de revue sur une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    payload = {
        "body": body,
        "event": "COMMENT",   # COMMENT, APPROVE ou REQUEST_CHANGES
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        review_id = resp.json().get("id")
        return f"Commentaire posté avec succès (review_id={review_id})."
    return f"Erreur GitHub {resp.status_code} : {resp.text}"


# ── Modèle via OpenRouter ──────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="anthropic/claude-sonnet-4-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://example.com",
        "X-Title": "TP Acte 1"
    },
)

# ── Prompt obligatoire pour AgentExecutor ──────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un revieweur de code Python senior. "
     "Commence TOUJOURS par récupérer les fichiers de la PR avec get_pr_files. "
     "Analyse ensuite le code : syntaxe, qualité, bonnes pratiques PEP8. "
     "Termine en postant un commentaire structuré sur GitHub avec post_review_comment. "
     "Format du commentaire : ## Revue\n### Problèmes\n### Suggestions\n### Verdict"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── Construction de l'agent legacy ────────────────────────────────────────────
tools = [get_pr_files, post_review_comment]
agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=6,
    return_intermediate_steps=True,   # ← clé de cet acte
    verbose=True,
)

# ── Invocation ────────────────────────────────────────────────────────────────
résultat = executor.invoke({
    "input": f"Fais une revue complète de la PR #{PR_NUMBER} et poste ton avis sur GitHub."
})

print("\n" + "="*60)
print("RÉPONSE FINALE :")
print(résultat["output"])
print("\n" + "="*60)
print(f"ÉTAPES INTERMÉDIAIRES : {len(résultat['intermediate_steps'])}")
for i, (action, observation) in enumerate(résultat["intermediate_steps"]):
    print(f"\n  Étape {i+1} — Tool : {action.tool}")
    print(f"  Arguments : {str(action.tool_input)[:80]}")
    print(f"  Résultat  : {str(observation)[:120]}...")
```

---

### Exercices

#### 1.1 — Observation de base

Lancez `python acte1.py` et répondez :

- Combien d'étapes intermédiaires l'agent a-t-il effectuées ?
- Dans quel ordre les tools ont-ils été appelés ? Était-ce l'ordre attendu ?
- Allez vérifier sur GitHub : le commentaire est-il bien apparu sur la PR ?

```python
# Explorez la structure
print(type(résultat["intermediate_steps"]))
print(type(résultat["intermediate_steps"][0]))

# Affichez le contenu brut du patch reçu par l'agent
action_0, obs_0 = résultat["intermediate_steps"][0]
print("\nPatch brut reçu par l'agent :")
print(obs_0[:500])
```

#### 1.2 — Ce que N8N ne montre pas

Dans N8N, le nœud AI Agent renvoie uniquement la réponse finale. Complétez ce tableau à partir de ce que vous observez dans `intermediate_steps` :

| Information | Visible dans N8N ? | Visible ici ? |
|---|:---:|:---:|
| Nom du tool appelé | ☐ | ☐ |
| Arguments passés au tool (dont le patch complet) | ☐ | ☐ |
| Réponse brute de l'API GitHub | ☐ | ☐ |
| Nombre d'itérations | ☐ | ☐ |
| Raisonnement interne du LLM | ☐ | ☐ |
| ID de la review postée | ☐ | ☐ |

#### 1.3 — Provoquer une erreur GitHub

Modifiez temporairement `GITHUB_TOKEN` pour qu'il soit invalide, relancez, et observez :

- L'agent récupère-t-il l'erreur HTTP 401 ou plante-t-il ?
- Que se passe-t-il si vous passez `handle_parsing_errors=True` à `AgentExecutor` ?
- Que se passe-t-il si vous le passez à `False` ?

Restaurez votre token correct avant de continuer.

#### 1.4 — Question de synthèse

N8N et cet `AgentExecutor` utilisent la même boucle ReAct.

**Citez deux choses** que vous pouvez faire ici et qui sont impossibles dans N8N.

**Citez une chose** que N8N facilite que vous devrez coder manuellement ici.

---

## Acte 2 — Migration vers `create_agent`

**Objectif :** migrer le revieweur vers l'API moderne et ajouter la suggestion inline via l'API GitHub Review.

### Comparaison côte à côte

Créez `acte2.py` en partant de cette structure :

```python
import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent   # ← nouveau

# ── Configuration GitHub ───────────────────────────────────────────────────────
GITHUB_TOKEN  = os.environ["GITHUB_TOKEN"]
GITHUB_REPO   = os.environ["GITHUB_REPO"]
PR_NUMBER     = int(os.environ["GITHUB_PR_NUMBER"])
HEADERS       = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE_URL      = f"https://api.github.com/repos/{GITHUB_REPO}"


# ── Tools GitHub ───────────────────────────────────────────────────────────────
@tool
def get_pr_files(pr_number: int) -> str:
    """Récupère la liste des fichiers modifiés dans une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur GitHub {resp.status_code} : {resp.text}"
    files = resp.json()
    result = []
    for f in files:
        result.append(
            f"Fichier : {f['filename']}\n"
            f"Patch   :\n{f.get('patch', '(pas de patch)')}"
        )
    return "\n---\n".join(result)


@tool
def post_review_comment(pr_number: int, body: str) -> str:
    """Poste un commentaire général de revue sur une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    payload = {"body": body, "event": "COMMENT"}
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return f"Commentaire posté (review_id={resp.json().get('id')})."
    return f"Erreur GitHub {resp.status_code} : {resp.text}"


@tool
def suggest_fix(pr_number: int, commit_id: str, path: str,
                line: int, suggestion: str, comment: str) -> str:
    """Poste une suggestion de correction inline sur une ligne précise de la PR.
    
    Args:
        pr_number  : numéro de la PR
        commit_id  : SHA du dernier commit de la PR (get_pr_info pour l'obtenir)
        path       : chemin du fichier (ex: calculator.py)
        line       : numéro de ligne dans le fichier
        suggestion : code de remplacement proposé (bloc ```suggestion```)
        comment    : explication de la suggestion
    """
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    body_text = f"{comment}\n\n```suggestion\n{suggestion}\n```"
    payload = {
        "commit_id": commit_id,
        "body": f"Suggestion sur {path}:{line}",   # body racine non vide (sinon 422)
        "event": "COMMENT",
        "comments": [
            {
                "path": path,
                "line": line,
                "body": body_text,
            }
        ]
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return f"Suggestion inline postée sur {path}:{line}."
    return f"Erreur GitHub {resp.status_code} : {resp.text}"


@tool
def get_pr_info(pr_number: int) -> str:
    """Récupère les métadonnées d'une PR : HEAD commit SHA, branche, auteur."""
    url = f"{BASE_URL}/pulls/{pr_number}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur GitHub {resp.status_code} : {resp.text}"
    data = resp.json()
    return (
        f"Titre    : {data['title']}\n"
        f"Auteur   : {data['user']['login']}\n"
        f"Branche  : {data['head']['ref']}\n"
        f"Commit   : {data['head']['sha']}\n"
        f"Statut   : {data['state']}"
    )


# ── Modèle ─────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="anthropic/claude-sonnet-4-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://example.com",
        "X-Title": "TP Acte 2"
    },
)

# ── TODO : créer l'agent avec create_agent ────────────────────────────────────
# Remplacez ce commentaire par le code de création de l'agent.
# Nommez impérativement votre variable `agent` : les exercices 2.2 et suivants
# la réutilisent telle quelle (agent.stream(...)).
# system_prompt : revieweur senior, utilise get_pr_info puis get_pr_files,
#                 poste un commentaire général ET au moins une suggestion inline

# ── TODO : invoquer l'agent ───────────────────────────────────────────────────
# Format d'entrée : {"messages": [{"role": "user", "content": "..."}]}
```

> ⚠️ **Piège GitHub à connaître — `422 Unprocessable Entity` sur `suggest_fix`**
> L'API GitHub exige que le paramètre `line` corresponde à une ligne **présente dans le diff de la PR**, et non à n'importe quelle ligne du fichier. Ici `calculator.py` est entièrement nouveau : toutes ses lignes figurent dans le diff, donc `suggest_fix` passe sans problème. Mais dès que vous testerez sur un fichier **modifié** (et non créé), viser une ligne hors du diff renverra une `422`. Retenez-le : c'est exactement le genre d'échec d'exécution que N8N masque et que `create_agent` vous laisse voir et gérer.

---

### Exercices

#### 2.1 — La migration minimale

Complétez `acte2.py` pour obtenir le même résultat qu'à l'Acte 1, **plus** au moins une suggestion inline.

Listez toutes les lignes que vous avez supprimées et celles que vous avez ajoutées :

| Supprimé (Acte 1) | Ajouté (Acte 2) |
|---|---|
| `from langchain_core.prompts import ...` | |
| `prompt = ChatPromptTemplate...` | |
| `create_tool_calling_agent(...)` | |
| `AgentExecutor(...)` | |

#### 2.2 — Observer la boucle avec le streaming

`create_agent` supporte le streaming natif. Une fois votre agent créé et nommé `agent` à l'étape 2.1, ajoutez ce bloc et observez la séquence :

```python
print("\n=== STREAMING PAS-À-PAS ===")
for step in agent.stream(
    {"messages": [{"role": "user",
                   "content": f"Revue complète de la PR #{PR_NUMBER} avec suggestions inline."}]},
    stream_mode="values",
):
    msg = step["messages"][-1]
    print(f"\n[{msg.__class__.__name__}]")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        # Affichez le nom du tool et les arguments clés
        for tc in msg.tool_calls:
            print(f"  → Tool : {tc['name']}")
            print(f"    Args : {str(tc['args'])[:120]}")
    else:
        print(f"  → {str(msg.content)[:100]}")
```

- Identifiez les types de messages qui apparaissent dans l'ordre.
- L'agent appelle-t-il `get_pr_info` avant `get_pr_files` ? Pourquoi est-ce important pour `suggest_fix` ?
- En combien d'étapes la boucle se termine-t-elle ?

#### 2.3 — Le system_prompt vs le prompt N8N

Testez les trois variantes et notez l'impact sur le comportement :

```python
# Variante A — instruction neutre
agent_A = create_agent(
    llm,
    tools=[get_pr_info, get_pr_files, post_review_comment, suggest_fix],
    system_prompt="Tu es un assistant."
)

# Variante B — revieweur exigeant
agent_B = create_agent(
    llm,
    tools=[get_pr_info, get_pr_files, post_review_comment, suggest_fix],
    system_prompt=(
        "Tu es un revieweur senior. "
        "Tu dois TOUJOURS appeler get_pr_info avant get_pr_files. "
        "Tu dois TOUJOURS poster au moins deux suggestions inline. "
        "Tu dois TOUJOURS conclure avec APPROUVÉ / CHANGEMENTS REQUIS."
    )
)

# Variante C — instruction contradictoire
agent_C = create_agent(
    llm,
    tools=[get_pr_info, get_pr_files, post_review_comment, suggest_fix],
    system_prompt=(
        "Tu es un assistant bienveillant. "
        "Ne critique jamais le code. Ne poste aucun commentaire négatif."
    )
)
```

- La variante C respecte-t-elle l'instruction face à du code manifestement défectueux ?
- Quelle différence avec N8N où ce contrôle n'existait pas ?

#### 2.4 — Changer de modèle sans changer de code

```python
# À tester : openai/gpt-4o-mini ou meta-llama/llama-3.3-70b-instruct
llm_b = ChatOpenAI(
    model="openai/gpt-4o-mini",   # ← seule ligne modifiée
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
agent_b = create_agent(
    llm_b,
    tools=[get_pr_info, get_pr_files, post_review_comment, suggest_fix],
    system_prompt="Tu es un revieweur de code Python senior."
)
```

- Le modèle moins puissant respecte-t-il l'ordre d'appel des tools (get_pr_info → get_pr_files) ?
- Les suggestions inline sont-elles aussi précises ?
- Le code `create_agent` a-t-il nécessité la moindre modification ?

---

### 2.5 — Human-in-the-loop : ce que `AgentExecutor` ne peut PAS faire

Jusqu'ici, la migration n'a rien apporté de visible : même revue, même résultat, juste moins de boilerplate. On pourrait conclure que `create_agent` est un simple sucre syntaxique. **C'est faux, et cet exercice le prouve.**

Mise en situation réaliste : un agent qui poste des commentaires sur une **vraie PR** ne doit pas publier sans validation. On veut que l'agent **s'arrête juste avant** chaque effet de bord GitHub (`post_review_comment`, `suggest_fix`), nous montre ce qu'il s'apprête à faire, et attende notre **approbation** pour reprendre. Les lectures (`get_pr_info`, `get_pr_files`) restent automatiques.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import textwrap

agent = create_agent(
    llm,
    tools=[get_pr_info, get_pr_files, post_review_comment, suggest_fix],
    system_prompt=(
        "Tu es un revieweur de code Python senior. "
        "Appelle get_pr_info puis get_pr_files, analyse le code, "
        "poste un commentaire général avec post_review_comment et au moins "
        "une suggestion inline avec suggest_fix."
    ),
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "get_pr_info":         False,                                  # lecture → auto
            "get_pr_files":        False,                                  # lecture → auto
            "post_review_comment": {"allowed_decisions": ["approve", "reject"]},
            "suggest_fix":         {"allowed_decisions": ["approve", "reject"]},
        },
        description_prefix="Action GitHub en attente d'approbation",
    )],
    checkpointer=InMemorySaver(),   # OBLIGATOIRE : sans persistance, pas de pause/reprise
)

config = {"configurable": {"thread_id": "review-pr-1"}}


def afficher_action(i, total, action):
    """Affiche une action GitHub en attente, de façon lisible (on ignore le
    champ `description`, redondant avec name + args)."""
    nom, args = action["name"], action["args"]
    print(f"\n  ── Action {i}/{total} : {nom} " + "─" * 20)
    if nom == "post_review_comment":
        print("  Type  : commentaire général de revue")
        print(textwrap.indent(args["body"].strip(), "    │ "))
    elif nom == "suggest_fix":
        print(f"  Cible : {args['path']} — ligne {args['line']}")
        print(f"  Motif : {args['comment'].splitlines()[0]}")
        print("  Suggestion proposée :")
        print(textwrap.indent(args["suggestion"], "    │ "))
    else:
        print(f"  Args  : {args}")


# 1er appel : l'agent lit la PR, rédige… puis SE FIGE avant de poster
résultat = agent.invoke(
    {"messages": [{"role": "user",
                   "content": f"Revue de la PR #{PR_NUMBER} avec suggestions inline."}]},
    config=config,
)

# Tant que l'agent est suspendu, on traite TOUTES les actions du batch.
# ⚠️ Le LLM appelle souvent plusieurs tools EN PARALLÈLE dans un même message ;
#    le middleware les regroupe en UNE interruption et exige une décision PAR
#    appel, dans le même ordre — d'où len(decisions) == len(action_requests).
while résultat.get("__interrupt__"):
    actions = résultat["__interrupt__"][0].value["action_requests"]
    print(f"\n⏸️  L'agent demande l'approbation de {len(actions)} action(s) GitHub.")

    decisions = []
    for i, action in enumerate(actions, start=1):
        afficher_action(i, len(actions), action)
        choix = input("  Approuver cette action ? [o/n] ").strip().lower()
        decisions.append({"type": "approve" if choix == "o" else "reject"})

    # On REPREND le graphe au point exact d'interruption, avec une décision/appel
    résultat = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )

print("\n" + "="*60)
print(résultat["messages"][-1].content)
```

> ⚠️ **Piège — `Number of human decisions (N) does not match number of hanging tool calls (M)`**
> Si le modèle propose plusieurs actions en parallèle (ici : 1 commentaire général + plusieurs `suggest_fix`), le middleware les **bundle dans une seule interruption** et attend **autant de décisions que d'appels en attente**, dans l'ordre. Fournir une seule décision lève une `ValueError`. C'est le revers direct du *batching parallèle* observé en 2.2 : `create_agent` regroupe les tool calls, donc le HITL regroupe aussi les approbations. Pour tout approuver d'un coup pendant une démo : `decisions = [{"type": "approve"}] * len(actions)`.

**Le contraste est le cœur de l'Acte 2 :**

| | Acte 1 — `AgentExecutor` | Acte 2 — `create_agent` + HITL |
|---|---|---|
| Moment où GitHub est appelé | dès que le LLM le décide, **sans recours** | l'agent **se fige sur le seuil** et te rend la main |
| Reprise après la décision | impossible (la boucle est déjà terminée) | `Command(resume=...)` reprend au point exact |
| État pendant l'attente | perdu (scratchpad transitoire) | snapshotté par le checkpointer |

**Pourquoi `AgentExecutor` ne peut structurellement pas faire ça.** `AgentExecutor.invoke()` est une **boucle Python fermée** : tu entres, la boucle ReAct tourne jusqu'au bout, tu sors. Son état intermédiaire n'est qu'un *scratchpad* (une string reformatée à chaque tour) qui vit dans la pile d'appels — il n'est ni nommé, ni adressable, ni persistable. Pour « mettre en pause avant un tool », il faudrait pouvoir **arrêter la boucle, sauvegarder tout l'état, et la relancer plus tard au même point** — ce qui suppose un état explicite et un checkpointer. `AgentExecutor` n'a ni l'un ni l'autre. Le `max_iterations` peut *arrêter* la boucle, mais jamais la *suspendre puis reprendre*.

#### Questions 2.5

- Combien d'actions le middleware vous a-t-il présentées en une seule interruption ? Reliez ce nombre au *batching parallèle* observé en 2.2 : pourquoi une seule décision provoque-t-elle une `ValueError` ?
- Lancez et **rejetez** (`n`) le `post_review_comment`. Que devient la suite de la boucle ? L'agent tente-t-il quand même les suggestions inline ?
- Tuez le script (`Ctrl-C`) pendant qu'il attend votre saisie, puis relancez-le **avec le même `thread_id`**. Que se passe-t-il ? Qu'est-ce que cela vous apprend sur le rôle du `checkpointer` ?
- En une phrase : pourquoi le `checkpointer` est-il *obligatoire* pour le HITL et pas pour l'agent de l'exercice 2.1 ?

---

### 2.6 — Rendre le graphe visible : implicite (`create_agent`) vs explicite (`StateGraph`)

Le HITL marche parce que `create_agent` **n'est pas une boucle, c'est un graphe LangGraph compilé.** Tu ne l'as pas dessiné — il est *préfabriqué* — mais il existe, et tu peux le voir :

```python
print(agent.get_graph().draw_mermaid())   # topologie en texte Mermaid
# (ou agent.get_graph().draw_mermaid_png() pour une image, en notebook)
```

Tu obtiens la topologie ReAct standard :

```
__start__ ──▶ agent ──▶ (tool calls ?) ──▶ tools ──▶ agent ──▶ … ──▶ __end__
                  └────────────── (sinon) ──────────────────────▶ __end__
```

Deux nœuds (`agent` = appel LLM, `tools` = exécution des tools), une arête conditionnelle qui boucle. Le HITL s'insère comme **point d'interruption sur l'arête `agent → tools`** : avant que le nœud `tools` ne s'exécute, le middleware lève un `interrupt`, le graphe sauvegarde son état et rend la main. C'est *ça*, la « logique de graphe » — et c'est ce qu'aucune boucle fermée ne peut offrir.

**La distinction à graver — trois niveaux, pas deux :**

| | `AgentExecutor` (Acte 1) | `create_agent` + HITL (Acte 2) | `StateGraph` (Acte 3 / cours) |
|---|---|---|---|
| Est-ce un graphe ? | **Non** — boucle Python fermée | **Oui, mais préfabriqué** (tu ne l'écris pas) | **Oui, explicite** (tu l'écris) |
| Qui définit la topologie | caché dans l'exécuteur | LangGraph (figée : `agent ⇄ tools`) | **toi** (`add_node` / `add_edge`) |
| État | scratchpad transitoire (string) | liste de messages dans le state, checkpointable | **TypedDict que tu définis** (+ reducers) |
| HITL / pause | impossible | **via middleware**, sans dessiner le graphe | via `interrupt()` dans tes propres nœuds |
| Persistance / reprise | non | oui (checkpointer) | oui (checkpointer) |
| Sur LangSmith | une trace plate d'appels | la trace d'une topologie **standard** que tu n'as pas conçue | **ta topologie sur-mesure**, nœuds nommés par toi |
| Quand l'utiliser | legacy, à éviter | agent unique + garde-fous (interrupt, retry, mémoire) | routage custom, branches, multi-agent |

C'est là toute la nuance que ce TP veut faire sentir : l'Acte 2 est un **graphe hybride**. Tu récupères les superpouvoirs du graphe (interruption, persistance, mémoire) **sans jamais écrire de nœuds ni d'arêtes** — LangGraph t'en fournit une topologie toute faite et tu te contentes d'y brancher du middleware. Tant que ton besoin tient dans la forme `un agent ⇄ ses tools`, ce niveau suffit, et c'est tant mieux : pas de graphe à maintenir.

Le jour où tu as besoin de **plusieurs nœuds que tu coordonnes toi-même** — un Revieweur *puis* un Suggéreur, avec un état partagé entre eux, un routage conditionnel, une validation entre les deux — la topologie préfabriquée ne suffit plus. Il te faut **dessiner le graphe**. C'est exactement le mur sur lequel l'Acte 3 va te faire foncer.

#### Questions 2.6

- Affichez le graphe de votre agent **sans** HITL (2.1), puis **avec** HITL (2.5). La topologie de base change-t-elle ? Où le middleware s'insère-t-il ?
- Dans le tableau ci-dessus, quelle ligne est selon vous la *vraie* raison de préférer `create_agent` à `AgentExecutor` ? (indice : ce n'est pas « moins de boilerplate »)
- Pré-formulez : qu'est-ce que `create_agent` ne pourra **pas** faire pour le pipeline Revieweur → Suggéreur de l'Acte 3, et pourquoi faudra-t-il un `StateGraph` ?

---

## Acte 3 — La frustration productive

**Objectif :** découvrir expérimentalement la limite structurelle de `create_agent` face à la coordination multi-agent — et comprendre pourquoi LangGraph StateGraph existe.

### Mise en situation

Vous devez construire un pipeline de revue en deux étapes :

- **Agent Revieweur** : récupère les fichiers de la PR, analyse le code, poste les commentaires de revue sur GitHub
- **Agent Suggéreur** : à partir de la revue, propose des suggestions inline sur les lignes problématiques

Le problème : le Suggéreur doit savoir **exactement quels problèmes ont été identifiés et sur quelles lignes** pour poster des suggestions cohérentes. Il a besoin du contexte complet du Revieweur — pas seulement de son texte de sortie.

> **Choix délibéré du TP :** le Suggéreur reçoit un outillage volontairement réduit (`get_pr_info` + `suggest_fix`, **sans** `get_pr_files`). Il ne peut donc pas relire le patch lui-même : il est contraint de se fier aux numéros de ligne présents dans la revue. Ce n'est pas un oubli — c'est précisément ce qui rend visible le silo mémoire entre les deux agents.

Créez `acte3.py` :

```python
import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# ── Configuration GitHub ───────────────────────────────────────────────────────
GITHUB_TOKEN  = os.environ["GITHUB_TOKEN"]
GITHUB_REPO   = os.environ["GITHUB_REPO"]
PR_NUMBER     = int(os.environ["GITHUB_PR_NUMBER"])
HEADERS       = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}"


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def get_pr_info(pr_number: int) -> str:
    """Récupère les métadonnées d'une PR : HEAD commit SHA, branche, auteur."""
    url = f"{BASE_URL}/pulls/{pr_number}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur {resp.status_code} : {resp.text}"
    data = resp.json()
    return (
        f"Titre  : {data['title']}\n"
        f"Auteur : {data['user']['login']}\n"
        f"Commit : {data['head']['sha']}\n"
        f"Branch : {data['head']['ref']}"
    )

@tool
def get_pr_files(pr_number: int) -> str:
    """Récupère la liste des fichiers modifiés dans une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return f"Erreur {resp.status_code} : {resp.text}"
    files = resp.json()
    result = []
    for f in files:
        result.append(
            f"Fichier : {f['filename']}\n"
            f"Patch   :\n{f.get('patch', '(pas de patch)')}"
        )
    return "\n---\n".join(result)

@tool
def post_review_comment(pr_number: int, body: str) -> str:
    """Poste un commentaire général de revue sur une Pull Request GitHub."""
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    payload = {"body": body, "event": "COMMENT"}
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return f"Commentaire posté (review_id={resp.json().get('id')})."
    return f"Erreur {resp.status_code} : {resp.text}"

@tool
def suggest_fix(pr_number: int, commit_id: str, path: str,
                line: int, suggestion: str, comment: str) -> str:
    """Poste une suggestion de correction inline sur une ligne précise de la PR."""
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    body_text = f"{comment}\n\n```suggestion\n{suggestion}\n```"
    payload = {
        "commit_id": commit_id,
        "body": f"Suggestion sur {path}:{line}",   # body racine non vide (sinon 422)
        "event": "COMMENT",
        "comments": [{"path": path, "line": line, "body": body_text}]
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    if resp.status_code in (200, 201):
        return f"Suggestion postée sur {path}:{line}."
    return f"Erreur {resp.status_code} : {resp.text}"


# ── Modèle ─────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="anthropic/claude-sonnet-4-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={"HTTP-Referer": "https://example.com", "X-Title": "TP Acte 3"},
)

# ── Agent Revieweur ────────────────────────────────────────────────────────────
agent_revieweur = create_agent(
    llm,
    tools=[get_pr_info, get_pr_files, post_review_comment],
    system_prompt=(
        "Tu es un revieweur Python senior. "
        "Appelle get_pr_info puis get_pr_files. "
        "Analyse le code et poste un commentaire structuré sur GitHub avec "
        "post_review_comment. Format : ## Revue\n### Problèmes (avec numéros "
        "de lignes exacts)\n### Verdict"
    ),
)

# ── Agent Suggéreur ───────────────────────────────────────────────────────────
agent_suggereur = create_agent(
    llm,
    tools=[get_pr_info, suggest_fix],   # volontairement SANS get_pr_files : il ne voit pas le patch
    system_prompt=(
        "Tu es un expert Python. À partir d'une revue de code, "
        "appelle get_pr_info pour obtenir le commit SHA, "
        "puis utilise suggest_fix pour poster des suggestions inline "
        "sur chaque ligne problématique identifiée dans la revue."
    ),
)

# ── Chaînage ──────────────────────────────────────────────────────────────────
print("=== ÉTAPE 1 : Agent Revieweur ===")
résultat_revieweur = agent_revieweur.invoke({
    "messages": [{"role": "user",
                  "content": f"Fais une revue de la PR #{PR_NUMBER}."}]
})
revue = résultat_revieweur["messages"][-1].content
print(revue)

print("\n=== ÉTAPE 2 : Agent Suggéreur ===")
résultat_suggereur = agent_suggereur.invoke({
    "messages": [{"role": "user",
                  "content": (
                      f"Voici la revue de la PR #{PR_NUMBER} :\n{revue}\n\n"
                      f"Poste des suggestions inline pour chaque problème identifié."
                  )}]
})
print(résultat_suggereur["messages"][-1].content)
```

---

### Exercices

#### 3.1 — Observer ce qui est transmis

Avant d'invoquer le Suggéreur, inspectez ce que le Revieweur a réellement produit :

```python
print("\n=== INSPECTION DU RÉSULTAT REVIEWEUR ===")
print(f"Nombre de messages dans l'historique : {len(résultat_revieweur['messages'])}")
for i, msg in enumerate(résultat_revieweur["messages"]):
    print(f"\n  Message {i} — type : {msg.__class__.__name__}")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  Tool calls : {[tc['name'] for tc in msg.tool_calls]}")
    print(f"  Contenu : {str(msg.content)[:150]}")

print(f"\n=== CE QUE LE SUGGÉREUR REÇOIT ===")
print(f"Seulement le dernier message : '{revue[:200]}...'")
print(f"\nCe que le Suggéreur NE VOIT PAS :")
print("  - Le commit SHA récupéré par get_pr_info (il devra le ré-appeler)")
print("  - Le patch brut récupéré par get_pr_files")
print("  - Les tool_calls intermédiaires du Revieweur")
print("  - Le review_id du commentaire déjà posté")
```

Complétez :

| Produit par le Revieweur | Transmis au Suggéreur ? |
|---|:---:|
| Texte de la réponse finale (la revue) | Oui |
| Commit SHA (récupéré via get_pr_info) | |
| Patch brut des fichiers | |
| Numéros de lignes exacts des problèmes | |
| review_id du commentaire posté | |
| Messages intermédiaires (tool calls) | |

**Conséquence concrète :** le Suggéreur doit rappeler `get_pr_info` pour obtenir le commit SHA — un appel API redondant qui aurait pu être évité avec un State partagé.

#### 3.2 — Tentative de contournement

Essayez de transmettre plus d'information au Suggéreur en construisant manuellement un contexte enrichi :

```python
# Tentative : extraire manuellement le commit SHA depuis les messages du Revieweur
commit_sha = None
for msg in résultat_revieweur["messages"]:
    # ToolMessage contient la réponse de get_pr_info
    if hasattr(msg, "content") and "Commit" in str(msg.content):
        for ligne in str(msg.content).split("\n"):
            if ligne.startswith("Commit"):
                commit_sha = ligne.split(":")[1].strip()
                break

contexte_enrichi = (
    f"PR #{PR_NUMBER} — commit SHA : {commit_sha or 'non trouvé'}\n\n"
    f"Revue produite :\n{revue}\n\n"
    f"Historique partiel du revieweur :\n"
    + "\n".join([
        f"  [{m.__class__.__name__}] {str(m.content)[:100]}"
        for m in résultat_revieweur["messages"]
    ])
)

résultat_v2 = agent_suggereur.invoke({
    "messages": [{"role": "user",
                  "content": (
                      f"Poste des suggestions inline.\nContexte : {contexte_enrichi}"
                  )}]
})
```

- Les suggestions inline sont-elles plus cohérentes avec la revue ?
- Quel est le problème de cette approche pour un cas de production ?
- Que se passerait-il si le Revieweur avait analysé 15 fichiers et posté 30 commentaires ?

#### 3.3 — Le diagnostic structurel

**1.** Dans le cours du matin, quel concept résoudrait le problème de transmission entre le Revieweur et le Suggéreur ?

**2.** Complétez ce schéma avec les termes vus ce matin :

```
Agent Revieweur (create_agent)       Agent Suggéreur (create_agent)
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ message history privé        │     │ message history vide          │
│ commit SHA (get_pr_info)     │──?──│ reçoit seulement :           │
│ patch brut (get_pr_files)    │     │   → texte final de la revue  │
│ review_id posté              │     │   → doit ré-appeler l'API    │
└──────────────────────────────┘     └──────────────────────────────┘

Ce pattern s'appelle        : ________________
La solution vue ce matin    : ________________
Le champ partagé idéal      : ________________  (TypedDict avec quels champs ?)
```

**3.** Parmi ces besoins, lesquels nécessitent un StateGraph plutôt que `create_agent` ?

| Besoin | `create_agent` suffit | Nécessite StateGraph |
|---|:---:|:---:|
| Reviewer un fichier unique en une passe | ☐ | ☐ |
| Revieweur + Suggéreur avec commit SHA partagé | ☐ | ☐ |
| Pause humaine avant de pousser les suggestions | ☐ | ☐ |
| Reprendre la revue après un timeout GitHub | ☐ | ☐ |
| Revieweur + Suggéreur + Formateur en parallèle | ☐ | ☐ |
| Réessayer suggest_fix si l'API GitHub renvoie 422 | ☐ | ☐ |

#### 3.4 — Conclusion

Rédigez en 5 lignes maximum :

> *« Dans quel cas utiliserais-je `create_agent` pour un pipeline de revue de code GitHub, et dans quel cas passerais-je directement au StateGraph ? »*

---

## Bilan du TP

| Concept | Acte | Ce que vous avez appris |
|---|:---:|---|
| Boucle ReAct visible | 1 | `intermediate_steps` expose ce que N8N cache |
| Tools GitHub réels | 1 | L'agent peut poster des reviews sur une vraie PR |
| Transfert de propriété de l'état | 2 | `create_agent` ne change pas la boucle ReAct, mais transfère la gestion de l'état (scratchpad, mémoire, contrôle) de ton code vers le graphe |
| Suggestion inline | 2 | L'API GitHub Review permet des `suggestion` blocks cliquables |
| Agnosticisme LLM | 2 | Un seul paramètre `model=` pour changer de fournisseur |
| Human-in-the-loop | 2 | Pause/reprise avant un effet de bord — impossible avec `AgentExecutor` (boucle fermée, pas de checkpointer) |
| Graphe implicite vs explicite | 2 | `create_agent` = graphe **préfabriqué** (`agent ⇄ tools`) ; HITL via middleware sans le dessiner. `StateGraph` = graphe **explicite** que tu écris |
| Silo mémoire | 3 | Le chaînage transmet seulement le texte final |
| Redondance API | 3 | Sans State partagé, le commit SHA est rappelé inutilement |
| Limite structurelle | 3 | `create_agent` = agent unique — le multi-agent nécessite StateGraph |

---

## Annexe — Structure des tools GitHub utilisés

| Tool | Endpoint GitHub | Méthode | Usage dans le TP |
|---|---|:---:|---|
| `get_pr_info` | `/repos/{owner}/{repo}/pulls/{pr}` | GET | Récupère le commit SHA |
| `get_pr_files` | `/repos/{owner}/{repo}/pulls/{pr}/files` | GET | Récupère les patchs |
| `post_review_comment` | `/repos/{owner}/{repo}/pulls/{pr}/reviews` | POST | Commentaire général |
| `suggest_fix` | `/repos/{owner}/{repo}/pulls/{pr}/reviews` | POST | Suggestion inline |

> **Note :** `suggest_fix` utilise la même route que `post_review_comment` mais avec un tableau `comments` contenant `path`, `line` et un bloc ` ```suggestion ``` `. GitHub rend ces suggestions cliquables et applicables en un clic dans l'interface PR. Attention : `line` doit pointer une ligne **présente dans le diff** de la PR, sinon l'API renvoie une `422`.

---

*Formation Agentique IA · Mai 2026*
