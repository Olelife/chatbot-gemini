from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

def setup_middlewares(app):

    origins = ["*"]

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_credentials=True,
        allow_headers=["*"],
    )

    # Header "x-username" obligatorio
    @app.middleware("http")
    async def require_username(request, call_next):
        username = request.headers.get("x-username")
        if not username:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing x-username header"}
            )
        request.state.username = username
        return await call_next(request)