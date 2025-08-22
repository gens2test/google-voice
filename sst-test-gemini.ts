import { GoogleGenAI, Type } from "@google/genai";
import type { TranscriptionSegment } from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable is not set.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.readAsDataURL(file);
  });
  return {
    inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
  };
};

export const transcribeAudio = async (
  file: File,
  history: TranscriptionSegment[] = []
): Promise<TranscriptionSegment[]> => {
  const model = 'gemini-2.5-flash';
  const audioPart = await fileToGenerativePart(file);

  try {
    // Step 1: Transcribe the audio.
    // To maximize stability and avoid internal server errors, we send *only* the audio data.
    // The model will infer the transcription task from the audio input.
    const transcriptionResult = await ai.models.generateContent({
      model: model,
      contents: [{ parts: [audioPart] }],
    });
    const rawTranscript = transcriptionResult.text.trim();

    if (!rawTranscript) {
      return [];
    }

    // Step 2: Use a separate, text-only call to structure the transcript and identify speakers.
    // This is more robust and allows us to reliably use the JSON schema feature.
    const historyContext = history.length > 0
      ? `You MUST be consistent with the speaker labels from the previous part of the conversation shown here:\n---\n${history.map(s => `${s.speaker}: ${s.dialogue}`).join('\n')}\n---`
      : 'Start by labeling the first speaker as "Speaker 1".';

    const structuringPrompt = `
      Analyze the following transcript and identify the different speakers.
      ${historyContext}
      Convert the transcript into a structured JSON format according to the provided schema.

      Transcript to analyze:
      ---
      ${rawTranscript}
      ---
    `;

    const responseSchema = {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          speaker: {
            type: Type.STRING,
            description: 'The identified speaker, e.g., "Speaker 1" or "Speaker 2".',
          },
          dialogue: {
            type: Type.STRING,
            description: 'The text spoken by that speaker in this segment.',
          },
        },
        required: ["speaker", "dialogue"],
      },
    };

    const structureResult = await ai.models.generateContent({
      model: model,
      contents: [{ parts: [{ text: structuringPrompt }] }],
      config: {
        responseMimeType: "application/json",
        responseSchema: responseSchema,
      },
    });

    const jsonText = structureResult.text.trim();

    try {
      const result = JSON.parse(jsonText) as TranscriptionSegment[];
      return result;
    } catch (parseError) {
      console.error("Failed to parse structured JSON from model:", parseError, "Raw Response:", jsonText);
      throw new Error("The model failed to return a valid JSON structure for the transcript.");
    }

  } catch (error) {
    console.error("Error transcribing audio:", error);
    let detailedMessage = "An unknown error occurred.";
    if (error instanceof Error) {
        detailedMessage = error.message;
        // Attempt to parse nested JSON error from the API for better readability.
        try {
            const errorObj = JSON.parse(detailedMessage);
            if (errorObj?.error?.message) {
                detailedMessage = errorObj.error.message;
            }
        } catch (e) {
            // It's not a JSON string, use the message as is.
        }
    }
    throw new Error(`Failed to transcribe audio: ${detailedMessage}`);
  }
};
