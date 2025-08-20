import { GoogleGenAI, Type } from "@google/genai";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

export const transcribeAudio = async (base64Audio: string, mimeType: string, language: string): Promise<{ transcription: string; dialect: string; standardThaiTranscription: string; }> => {
  try {
    const audioPart = {
      inlineData: {
        mimeType: mimeType,
        data: base64Audio,
      },
    };
    const textPart = {
      text: `You are an expert linguist. The user is providing an audio file expected to be in ${language}. Your task is to:
1.  **Transcribe** the audio accurately.
2.  **Identify Dialect**: If the language is Thai, identify the specific regional dialect (e.g., Northern, Northeastern, Southern, Central). For other languages, or if you cannot determine the dialect, respond with "Unknown".
3.  **Convert to Standard Thai**: If the language is Thai and a regional dialect is detected, convert the transcription to Standard Central Thai. For example, convert "ช่วยอู้จะจานหน่อยได้ก่อ" (Northern) to "ช่วยพูดช้าๆหน่อยได้ไหม". If the audio is already in Standard Thai, or not in Thai, this field should contain the original transcription.`
    };

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{ parts: [audioPart, textPart] }],
      config: {
          responseMimeType: "application/json",
          responseSchema: {
              type: Type.OBJECT,
              properties: {
                  transcription: {
                      type: Type.STRING,
                      description: "The transcribed text from the audio."
                  },
                  dialect: {
                      type: Type.STRING,
                      description: "The identified Thai dialect (e.g., Southern Thai) or 'Unknown'."
                  },
                  standardThaiTranscription: {
                      type: Type.STRING,
                      description: "The transcription converted to Central Thai (Standard Thai). If no conversion is needed, this will be the same as the transcription."
                  }
              }
          }
      }
    });

    const jsonResponse = JSON.parse(response.text);
    return {
        transcription: jsonResponse.transcription || "",
        dialect: jsonResponse.dialect || "Unknown",
        standardThaiTranscription: jsonResponse.standardThaiTranscription || jsonResponse.transcription || "",
    };
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw new Error("Failed to transcribe audio with Gemini API.");
  }
};
