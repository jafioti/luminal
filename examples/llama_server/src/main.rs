use axum::{
    extract::Extension,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::sync::{Arc, Mutex, RwLock};
use tokio::net::TcpListener;

mod chat;
mod llama;

use crate::llama::setup::Model;
use chat::{respond_chat_request, ChatRequest, ChatResponse};

#[tokio::main]
async fn main() {
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
    // let mut model = model.lock().unwrap(); // Acquire a write lock

    // let response = respond_chat_request(&mut *model, payload).await;
    // (StatusCode::OK, Json(response))
    (
        StatusCode::OK,
        Json(ChatResponse {
            id: "".to_string(),
            created: 1,
            object: "chat.completion".to_string(),
            model: "meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
            choices: vec![chat::Choice {
                index: 0,
                message: chat::Message {
                    role: chat::Role::Assistant,
                    content: "".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: chat::Usage {
                total_tokens: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
            },
        }),
    )
}
