use std::sync::Arc;

use axum::{
    extract::Extension,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
mod chat;
mod llama;

use chat::{respond_chat_request, ChatRequest, ChatResponse};

use crate::llama::setup::Model;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model = Arc::new(Model::setup());

    let app = Router::new()
        .route("/", get(root))
        .route("/chat/completions", post(chat_completions))
        .layer(Extension(model));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, World!"
}

async fn chat_completions(
    Extension(model): Extension<Arc<Model>>,
    Json(payload): Json<ChatRequest>,
) -> (StatusCode, Json<ChatResponse>) {
    (
        StatusCode::OK,
        Json(respond_chat_request(&model, payload).await),
    )
}
