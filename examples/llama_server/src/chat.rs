use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// src/chat.rs
use crate::llama::setup::Model; // Import the Model struct

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
}

#[derive(Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Debug)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "user")]
    User,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Respond to chat request
pub async fn respond_chat_request(model: &Model, request: ChatRequest) -> ChatResponse {
    let created = Utc::now().timestamp();
    let raw_uuid = Uuid::new_v4();
    let id = format!("chatcmpl-{}", raw_uuid);
    let response = ChatResponse {
        id,
        created,
        object: "chat.completion".to_string(),
        model: "gpt-3.5-turbo-0125".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
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

    response
}
