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
    # model="anthropic/claude-sonnet-4-5",
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://example.com",
        "X-Title": "TP Acte 2"
    },
)

tools = [get_pr_info,get_pr_files, post_review_comment,suggest_fix]

newAgent = create_agent(
    model=llm,
    tools=tools,
    # system_prompt="revieweur senior, utilise get_pr_info puis get_pr_files,"
    #               "poste un commentaire général ET au moins une suggestion inline"
    # system_prompt="Tu es un assistant."
    # system_prompt=(
    #     "Tu es un revieweur senior. "
    #     "Tu dois TOUJOURS appeler get_pr_info avant get_pr_files. "
    #     "Tu dois TOUJOURS poster au moins deux suggestions inline. "
    #     "Tu dois TOUJOURS conclure avec APPROUVÉ / CHANGEMENTS REQUIS."
    # )
    # system_prompt=(
    #     "Tu es un assistant bienveillant. "
    #     "Ne critique jamais le code. Ne poste aucun commentaire négatif."
    # )
    system_prompt="Tu es un revieweur de code Python senior."
)

# result = newAgent.invoke({
#     "messages": [
#         {"role": "user", "content": f"Fais la revue de la PR {PR_NUMBER}"}
#     ]
# })

for step in newAgent.stream(
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