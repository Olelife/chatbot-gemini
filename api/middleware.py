from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

def setup_middlewares(app):

    origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:6006",
        "https://portal.dev.ole-sp.com.mx",
        "https://portal.stg.ole-sp.com.mx",
        "https://portal.ole-sp.com.mx",
        "https://portal.olelife.com.br",
    ]

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_credentials=True,
        allow_headers=["*"],
    )

    protected_paths = [
        "/ask",
        "/ask/",
    ]

    # Header "x-username" obligatorio
    @app.middleware("http")
    async def require_username(request, call_next):
        path = request.url.path

        if not any(path.startswith(p) for p in protected_paths):
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        username = request.headers.get("x-username")

        slack_paths = ["/slack/command", "/ask/slack"]

        if request.url.path.lower() in slack_paths:
            return await call_next(request)

        if not username:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing x-username header"}
            )
        request.state.username = username
        return await call_next(request)