use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
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
pub async fn respond_chat_request(request: ChatRequest) -> ChatResponse {
    let response = ChatResponse {
        id: "chatcmpl-9J4pxGD1wK3SQzmt70sspIuaZvFaP".to_string(),
        object: "chat.completion".to_string(),
        created: 1714333853,
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
