import { GoogleGenAI, Type } from '@google/genai';
import type { AiResponse } from '../types';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not set. The application cannot function without it.");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const responseSchema = {
    type: Type.OBJECT,
    properties: {
        languageCode: {
            type: Type.STRING,
            description: "The BCP-47 language code of the user's input text (e.g., 'en-US', 'es-MX', 'fr-FR', 'th-TH').",
        },
        response: {
            type: Type.STRING,
            description: "A helpful and natural-sounding response to the user's query, delivered in the same language and dialect as the input. For example, if the input is in Thai, the response must also be in Thai.",
        },
    },
    required: ['languageCode', 'response'],
};

const MAX_RETRIES = 4;
const INITIAL_BACKOFF_MS = 1500;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const getAiResponse = async (userTranscript: string): Promise<AiResponse | null> => {
    if (!userTranscript) return null;

    let lastError: any = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const result = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: `You are a helpful voice assistant. Your primary goal is to answer the user's question accurately and conversationally. First, identify the language and dialect of the user's text. Then, provide a helpful and natural-sounding response to their query in that same language and dialect. Format your output as a JSON object that strictly adheres to the provided schema. User's text: "${userTranscript}"`,
                config: {
                    responseMimeType: 'application/json',
                    responseSchema: responseSchema,
                    temperature: 0.7,
                },
            });

            const jsonString = result.text.trim();
            const parsedResponse: AiResponse = JSON.parse(jsonString);
            return parsedResponse; // Success

        } catch (error) {
            lastError = error;
            // The error from the API might not have a standard `message` property.
            // Stringify the entire error to reliably check for rate limit indicators.
            const errorJsonString = JSON.stringify(error);
            
            if (errorJsonString.includes('429') || errorJsonString.includes('RESOURCE_EXHAUSTED')) {
                if (attempt < MAX_RETRIES) {
                    const backoffTime = INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
                    console.warn(`Rate limit hit. Retrying in ${backoffTime}ms... (Attempt ${attempt}/${MAX_RETRIES})`);
                    await sleep(backoffTime);
                }
            } else {
                // Not a rate limit error, fail fast
                break;
            }
        }
    }
    
    // If all retries fail, or a non-retriable error occurred
    console.error("Error calling Gemini API after retries:", lastError);
    const finalErrorJsonString = JSON.stringify(lastError);
    
    if (finalErrorJsonString.includes('429') || finalErrorJsonString.includes('RESOURCE_EXHAUSTED')) {
        return {
            languageCode: 'en-US',
            response: "I'm currently experiencing high demand and couldn't process your request. Please try again in a few moments."
        };
    }

    return {
        languageCode: 'en-US',
        response: "I'm sorry, I seem to be having trouble connecting. Please try again in a moment."
    };
};
