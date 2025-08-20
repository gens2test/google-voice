
import { GoogleGenAI } from "@google/genai";
import { Dialect } from '../types';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const translateToDialect = async (text: string, dialect: Dialect): Promise<string> => {
  try {
    const prompt = `Translate the following standard Thai text into the ${dialect} dialect.
    Provide ONLY the translated text, with no extra explanations, labels, or formatting.
    For example, if the input is 'พูดช้าๆ' and the dialect is Northern, the output should be 'อู้จ้าๆ'.
    
    Text to translate: "${text}"`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        temperature: 0.3,
      }
    });
    
    const translatedText = response.text.trim();
    
    if (!translatedText) {
        throw new Error("Received an empty response from the AI. The model might be unable to translate this phrase.");
    }
    
    return translatedText;

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    if (error instanceof Error && error.message.includes('API key not valid')) {
       throw new Error("The provided API key is not valid. Please check your configuration.");
    }
    throw new Error("Failed to translate text. Please try again later.");
  }
};
