use axum::{
    extract::Extension,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;

mod chat;
mod llama;

use crate::llama::setup::Model;
use chat::{respond_chat_request, ChatRequest, ChatResponse};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model = Arc::new(RwLock::new(Model::setup()));

    let app = Router::new()
        .route("/", get(root))
        .route("/chat/completions", post(chat_completions))
        .layer(Extension(model));

    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, World!"
}

async fn chat_completions(
    Extension(model): Extension<Arc<RwLock<Model>>>,
    Json(payload): Json<ChatRequest>,
) -> (StatusCode, Json<ChatResponse>) {
    let mut model = model.write().unwrap(); // Acquire a write lock

    let response = respond_chat_request(&mut *model, payload).await;
    (StatusCode::OK, Json(response))
}
