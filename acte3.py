import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import textwrap


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
    # model="anthropic/claude-sonnet-4-5",
    model="anthropic/claude-sonnet-4-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://example.com",
        "X-Title": "TP Acte 2"
    },
)

tools = [get_pr_info,get_pr_files, post_review_comment,suggest_fix]

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
print(agent.get_graph().draw_mermaid())   # topologie en texte Mermaid


config = {"configurable": {"thread_id": f"review-pr-{PR_NUMBER}"}}


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