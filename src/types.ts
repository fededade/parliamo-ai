export interface TranscriptItem {
  id: string;
  sender: 'user' | 'model';
  type: 'text' | 'image' | 'action';
  text?: string;
  image?: string;
  isComplete: boolean;
  actionUrl?: string;
  actionLabel?: string;
  actionIcon?: string;
}

export interface AssistantConfig {
  userName: string;
  gender: string;
  age: string;
  hairColor: string;
  eyeColor: string;
  skinTone: string;
  bodyType: string;
  physicalTraits: string;
  personality: string;
  name: string;
  biography: string;
  visualPrompt: string;
  voicePitch: number;
  voiceSpeed: number;
  voiceEnergy: number;
  voiceTone: number;
}
