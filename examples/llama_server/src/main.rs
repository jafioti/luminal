use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
mod chat;
use chat::{respond_chat_request, ChatRequest, ChatResponse};

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt::init();

    // build our application with routes
    let app = Router::new()
        .route("/", get(root))
        .route("/chat/completions", post(chat_completions));

    // run our app with hyper
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with "Hello, World!"
async fn root() -> &'static str {
    "Hello, World!"
}

// mock chat completions handler
async fn chat_completions(Json(payload): Json<ChatRequest>) -> (StatusCode, Json<ChatResponse>) {
    (StatusCode::OK, Json(respond_chat_request(payload).await))
}
