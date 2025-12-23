export interface TranscriptItem {
  id: string;
  sender: 'user' | 'model';
  type: 'text' | 'image' | 'action'; // New field to strictly separate rendering logic
  text?: string;
  image?: string; // Base64 data url
  isComplete: boolean;
  // Action Button Fields
  actionUrl?: string;
  actionLabel?: string;
  actionIcon?: 'mail' | 'message-circle';
}

export interface AudioVisualizerProps {
  isPlaying: boolean;
  volume: number; // 0 to 1
}

export interface AssistantConfig {
  gender: string;
  age: string;
  hairColor: string;
  eyeColor: string;
  skinTone: string;
  physicalTraits: string;
  personality: string;
  // Generated fields
  name?: string; // AI Name (can be manually set or generated)
  userName?: string; // User's Name
  biography?: string;
  visualPrompt?: string; // The "Visual DNA" for consistency
  // Voice Modulation
  voicePitch?: number; // Detune in cents (-200 to 200)
  voiceSpeed?: number; // Playback rate (0.9 to 1.1)
}