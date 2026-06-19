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
BASE_URL      = f"https://api.github.com/repos/OMIMGitHub/IA_Training"


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
    handle_parsing_errors=True,
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
