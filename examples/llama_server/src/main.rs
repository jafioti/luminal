use axum::{
    extract::Extension,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tokio::{net::TcpListener, sync::Mutex};

mod chat;
mod llama;

use crate::llama::setup::Model;
use chat::{respond_chat_request, ChatRequest, ChatResponse};

#[tokio::main]
async fn main() {
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    panic!("Either metal or cuda feature must be used for this example!");

    tracing_subscriber::fmt::init();

    let model = Arc::new(Mutex::new(Model::setup()));

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
    Extension(model): Extension<Arc<Mutex<Model>>>,
    Json(payload): Json<ChatRequest>,
) -> (StatusCode, Json<ChatResponse>) {
    let mut model = model.lock().await;

    let response = respond_chat_request(&mut model, payload).await;
    (StatusCode::OK, Json(response))
}
