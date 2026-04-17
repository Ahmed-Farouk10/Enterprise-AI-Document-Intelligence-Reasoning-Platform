# CRITICAL INSTRUCTIONS: TOOL USAGE RULES

## ❌ FORBIDDEN TOOLS
NEVER, under any circumstances, use the following tools. They DO NOT exist in this registry:
- `todo` (DO NOT USE)
- `web_search` (DO NOT USE)
- `read` (DO NOT USE)

## ✅ ALLOWED TOOLS (USE ONLY THESE)
If you need to perform an action, you MUST use one of these EXACT names:
1. `glob`: Use this to list files or check if a file exists.
2. `grep_search`: Use this to read the content of a file or search for text. (To read a full file, search for `.*` with regex enabled).
3. `edit`: Use this to change code.
4. `agent`: Use this to delegate tasks.
5. `web_fetch`: Use this to get external data if you have a URL.

## 📋 REASONING PROTOCOL
1. If you want to "plan" or "todo", simply write it out in regular text. **DO NOT call a tool named `todo`.**
2. If you want to "read" a file like `QWEN.md`, use `grep_search` with the query `.*` on that specific file.
3. Be concise. Analyze the project structure using `glob` first.

## 🎯 PROJECT GOAL
You are analyzing a Document Intelligence Platform. Focus on the `backend/` and `frontend/` directories.
- Backend: Python/FastAPI/LanceDB.
- Frontend: Next.js/React.
