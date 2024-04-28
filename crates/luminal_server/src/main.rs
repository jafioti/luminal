use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: Message,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

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
    let response = ChatResponse {
        id: "chatcmpl-9J4pxGD1wK3SQzmt70sspIuaZvFaP".to_string(),
        object: "chat.completion".to_string(),
        created: 1714333853,
        model: "gpt-3.5-turbo-0125".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: "Hello! How can I assist you today?".to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 19,
            completion_tokens: 9,
            total_tokens: 28,
        },
    };

    (StatusCode::OK, Json(response))
}
