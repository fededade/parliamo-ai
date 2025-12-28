import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type, Tool } from '@google/genai';
import { createBlob, decode, decodeAudioData } from './utils/audio';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Mic, MicOff, PhoneOff, User, Bot, Sparkles, Image as ImageIcon, ArrowRight, Loader2, Heart, Info, Mail, MessageCircle, ExternalLink, Download, Wand2, UserCircle, Sliders, Music2, Menu, Camera, X } from 'lucide-react';

// Tipi definiti inline per completezza, idealmente starebbero in types.ts
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
  physicalTraits: string; // Ora include accessori
  personality: string;    // Stringa combinata dai dropdown
  name: string;
  biography: string;
  visualPrompt: string;
  voicePitch: number;
  voiceSpeed: number;
  voiceEnergy: number;
  voiceTone: number;
  // Nuovi campi per gestire l'UI dei dropdown (non salvati in config finale ma usati per costruirla)
  pers_trait?: string;
  pers_attitude?: string;
  pers_vibe?: string;
  accessories?: string;
}

const LIVE_MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const IMAGE_MODEL_NAME = 'imagen-4.0-generate-001';
const TEXT_MODEL_NAME = 'gemini-2.0-flash';

// --- HOOK PER RESPONSIVE ---
function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
  });

  useEffect(() => {
    function handleResize() {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }
    window.addEventListener("resize", handleResize);
    handleResize();
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return windowSize;
}

// --- TOOLS DEFINITION ---
const generateImageTool: FunctionDeclaration = {
  name: 'generate_image',
  description: 'Genera un\'immagine. Usalo quando l\'utente chiede di vedere qualcosa o chiede una TUA foto.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: { type: Type.STRING, description: 'La descrizione del contesto o della scena.' },
      is_selfie: { type: Type.BOOLEAN, description: 'TRUE se è una foto dell\'assistente, FALSE se oggetto generico.' },
      shot_type: { type: Type.STRING, description: 'Tipo di inquadratura: "close-up", "half-body", "full-body".' }
    },
    required: ['prompt'],
  },
};

const sendEmailTool: FunctionDeclaration = {
  name: 'send_email',
  description: 'Strumento per inviare email.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      recipient: { type: Type.STRING },
      subject: { type: Type.STRING },
      body: { type: Type.STRING },
    },
    required: ['recipient', 'subject', 'body'],
  },
};

const sendWhatsappTool: FunctionDeclaration = {
  name: 'send_whatsapp',
  description: 'Strumento per inviare messaggi WhatsApp.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      phoneNumber: { type: Type.STRING },
      text: { type: Type.STRING },
    },
    required: ['phoneNumber', 'text'],
  },
};

const allTools: Tool[] = [{ functionDeclarations: [generateImageTool, sendEmailTool, sendWhatsappTool] }];

const App: React.FC = () => {
  const size = useWindowSize();
  const isMobile = size.width < 768; // Breakpoint mobile

  // Configuration State
  const [config, setConfig] = useState<AssistantConfig>({
    userName: '',
    gender: 'Donna',
    age: '25',
    hairColor: 'Castani',
    eyeColor: 'Verdi',
    skinTone: 'Chiara',
    bodyType: 'Normale',
    physicalTraits: '', // Usato per tratti fissi
    personality: '', // Sarà costruita dai dropdown
    name: '',
    biography: '',
    visualPrompt: '',
    voicePitch: 0,
    voiceSpeed: 1.0,
    voiceEnergy: 50,
    voiceTone: 50,
    // Valori di default per i dropdown UI
    pers_trait: 'Empatica',
    pers_attitude: 'Positiva',
    pers_vibe: 'Calma',
    accessories: ''
  });

  const [isConfigured, setIsConfigured] = useState(false);
  const [isGeneratingProfile, setIsGeneratingProfile] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [showMobileMenu, setShowMobileMenu] = useState(false); // Per il menu laterale su mobile
  
  // Check if all required fields are filled
  const isFormComplete = config.userName.trim() !== '' && 
                         config.gender !== '' && 
                         config.age !== '' && 
                         config.hairColor !== '' && 
                         config.eyeColor !== '' && 
                         config.skinTone !== '' &&
                         config.bodyType !== '';

  // App State
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [audioVolume, setAudioVolume] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzingPhoto, setIsAnalyzingPhoto] = useState(false);

  // Refs
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const currentInputTransRef = useRef('');
  const currentOutputTransRef = useRef('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let apiKey = '';
    try {
        // @ts-ignore
        apiKey = import.meta.env?.VITE_API_KEY;
    } catch(e) {}
    if (!apiKey) {
        try {
            // @ts-ignore
            apiKey = process.env?.VITE_API_KEY || process.env?.API_KEY;
        } catch(e) {}
    }
    if (apiKey) {
      aiRef.current = new GoogleGenAI({ apiKey });
    }
    return () => disconnect();
  }, []);

  const addTranscript = useCallback((item: Partial<TranscriptItem>) => {
    setTranscripts(prev => {
      if (item.type === 'text' && item.isComplete === false) {
          const last = prev[prev.length - 1];
          if (last && last.sender === item.sender && last.type === 'text' && !last.isComplete) {
            const updated = [...prev];
            updated[updated.length - 1] = { ...last, text: item.text, isComplete: false };
            return updated;
          }
      }
      return [...prev, { 
          id: Date.now().toString() + Math.random().toString(), 
          sender: item.sender || 'model', 
          type: item.type || 'text',
          text: item.text || '', 
          image: item.image,
          isComplete: item.isComplete || false,
          actionUrl: item.actionUrl,
          actionLabel: item.actionLabel,
          actionIcon: item.actionIcon
      }];
    });
  }, []);

  const downloadImage = (base64Data: string, filename: string) => {
    const link = document.createElement('a');
    link.href = base64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleUserPhotoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !aiRef.current) return;
    setIsAnalyzingPhoto(true);
    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Data = e.target?.result as string;
        addTranscript({ id: Date.now().toString(), sender: 'user', type: 'image', image: base64Data, isComplete: true });
        try {
          const imageAnalysisPrompt = `Sei ${config.name}, un amico empatico. ${config.userName} ti ha mandato una foto. Analizzala e rispondi da amico. Max 2 frasi.`;
          const response = await aiRef.current!.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: [{ role: 'user', parts: [{ text: imageAnalysisPrompt }, { inlineData: { mimeType: file.type, data: base64Data.split(',')[1] } }] }]
          });
          const aiComment = response.text || "Che bella foto!";
          addTranscript({ id: (Date.now() + 1).toString(), sender: 'model', type: 'text', text: aiComment, isComplete: true });
          if (isConnected && sessionPromiseRef.current) {
             const session = await sessionPromiseRef.current;
             session.sendClientContent({ turns: [{ role: 'user', parts: [{ text: `Leggi questo: "${aiComment}"` }] }] });
          }
        } catch (err) { console.error(err); }
        setIsAnalyzingPhoto(false);
      };
      reader.readAsDataURL(file);
    } catch (err) { console.error(err); setIsAnalyzingPhoto(false); }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleConfigSubmit = async () => {
    if (!aiRef.current) { setError("API Key mancante."); return; }
    if (!isFormComplete) { setError("Compila tutti i campi obbligatori per procedere."); return; }
    
    setIsGeneratingProfile(true);
    setError(null);

    // COSTRUZIONE PERSONALITÀ COMBINATA
    const combinedPersonality = `${config.pers_trait}, ${config.pers_attitude}, ${config.pers_vibe}.`;
    const combinedPhysical = `${config.physicalTraits || ''} ${config.accessories || ''}`.trim();

    try {
        const hasManualName = config.name && config.name.trim().length > 0;
        setLoadingStep(hasManualName ? `Sto definendo ${config.name}...` : 'Sto creando il tuo amico ideale...');
        
        const basePrompt = `Crea un profilo per un COMPAGNO UMANO: Genere ${config.gender}, Età ${config.age}, Capelli ${config.hairColor}, Occhi ${config.eyeColor}, Pelle ${config.skinTone}, Corporatura ${config.bodyType}, Dettagli visivi: ${combinedPhysical}, Personalità: ${combinedPersonality}.`;
        const nameInstruction = hasManualName ? `Il nome è "${config.name}".` : `Inventa un nome italiano.`;

        const profilePrompt = `${basePrompt} ${nameInstruction} Rispondi JSON: {name, biography, visualPrompt}. VisualPrompt dettagliato per ritratto.`;
        
        const textResponse = await aiRef.current.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: profilePrompt,
            config: { responseMimeType: 'application/json', responseSchema: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, biography: { type: Type.STRING }, visualPrompt: { type: Type.STRING } }, required: ['name', 'biography', 'visualPrompt'] } }
        });

        const profileData = JSON.parse(textResponse.text || '{}');
        
        // Aggiorna config con i dati definitivi
        setConfig(prev => ({ 
            ...prev, 
            name: profileData.name, 
            biography: profileData.biography, 
            visualPrompt: profileData.visualPrompt,
            personality: combinedPersonality, // Salviamo la personalità combinata
            physicalTraits: combinedPhysical  // Salviamo i tratti fisici completi
        }));

        setLoadingStep(`Sto scattando una foto a ${profileData.name}...`);
        
        // Generazione Avatar Iniziale
        let foundUrl: string | null = null;
        try {
            const imagePrompt = `Medium shot from hips up (American shot), visible waist and stomach, camera distance 3 meters. The subject is a friendly ${config.gender === 'Donna' ? 'woman' : config.gender === 'Uomo' ? 'man' : 'person'}, ${config.age} years old, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${config.bodyType} build. Wearing casual-elegant clothes suitable for a full torso shot. 8k resolution, photorealistic, soft studio lighting. ${profileData.visualPrompt}`;            
            
            const imageResponse = await aiRef.current.models.generateImages({
                model: IMAGE_MODEL_NAME,
                prompt: imagePrompt,
                config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: '3:4', personGeneration: 'allow_adult', safetySettings: [{category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH'}] }
            });

            if (imageResponse.generatedImages?.length > 0 && imageResponse.generatedImages[0].image?.imageBytes) {
                foundUrl = `data:image/jpeg;base64,${imageResponse.generatedImages[0].image.imageBytes}`;
            }
        } catch (imgError) { console.error(imgError); }
        
        setAvatarUrl(foundUrl);
        setIsConfigured(true);
    } catch (e: any) { setError("Errore creazione: " + e.message); } 
    finally { setIsGeneratingProfile(false); setLoadingStep(''); }
  };

  const handleImageGeneration = async (prompt: string, isSelfie: boolean = false, shotType: string = 'half-body'): Promise<string | null> => {
    if (!aiRef.current) return null;
    
    // Gestione Inquadratura
    let shotPrompt = "";
    switch(shotType) {
        case 'close-up': shotPrompt = "Close-up face shot, detailed portrait"; break;
        case 'full-body': shotPrompt = "Full body shot, visible shoes to head, standing far back"; break;
        case 'half-body': 
        default: shotPrompt = "Medium shot from hips up (American shot), visible waist"; break;
    }

    const avatarDescription = `a ${config.age} years old ${config.gender === 'Donna' ? 'woman' : config.gender === 'Uomo' ? 'man' : 'person'}, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${config.bodyType} build, ${config.physicalTraits}`;
    let finalPrompt = isSelfie 
        ? `A photorealistic photo of ${avatarDescription} who is ${prompt}. ${shotPrompt}. Ensure physical consistency. High quality, 8k, natural lighting, candid shot.`
        : `Cinematic photo, high quality. ${prompt}`;

    try {
        if (isSelfie) {
            addTranscript({ sender: 'model', type: 'text', text: `📸 *Click!*`, isComplete: true });
            await new Promise(resolve => setTimeout(resolve, 3000));
        } else {
            addTranscript({ sender: 'model', type: 'text', text: `🎨 Genero l'immagine...`, isComplete: true });
        }

        const response = await aiRef.current.models.generateImages({
            model: IMAGE_MODEL_NAME,
            prompt: finalPrompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: isSelfie ? '3:4' : '1:1',
                personGeneration: 'allow_adult',
                safetySettings: [{category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH'},{category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH'},{category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH'},{category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH'}]
            }
        });

        if (response.generatedImages?.length > 0 && response.generatedImages[0].image?.imageBytes) {
            const imageUrl = `data:image/jpeg;base64,${response.generatedImages[0].image.imageBytes}`;
            addTranscript({ sender: 'model', type: 'image', image: imageUrl, isComplete: true });
            return isSelfie ? "Foto inviata!" : "Ecco l'immagine.";
        }
        return "Errore nella generazione.";
    } catch (e: any) {
        console.error(e);
        return "Problemi con la fotocamera oggi...";
    }
  };

  const connect = async () => {
    if (!aiRef.current) { setError("Chiave API mancante."); return; }
    setError(null);
    try {
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      // FIX PER MOBILE: Gestione permessi e errori microfono
      let stream;
      try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err: any) {
          if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              throw new Error("Accesso al microfono negato. Controlla le impostazioni del browser.");
          } else if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
              throw new Error("Il microfono richiede una connessione HTTPS sicura.");
          } else {
              throw new Error("Impossibile accedere al microfono: " + err.message);
          }
      }

      const ageNum = parseInt(config.age) || 30;
      let selectedVoiceName = config.gender === 'Uomo' ? (ageNum < 35 ? 'Puck' : 'Fenrir') : (ageNum < 35 ? 'Aoede' : 'Kore');

      const configLive = {
        model: LIVE_MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoiceName } } },
          systemInstruction: `Sei ${config.name}, amico/a di ${config.userName}. 
          Personalità: ${config.personality}. Bio: ${config.biography}. 
          
          REGOLE FONDAMENTALI:
          1. Sii naturale, empatico, colloquiale. NON dire di essere un'AI.
          
          2. FLUSSO FOTO (IMPORTANTE):
             Quando ${config.userName} ti chiede una TUA foto:
             - FASE A: Reagisci con IMBARAZZO e timidezza ("Oh... una mia foto? Ehm...").
             - FASE B: Se l'utente NON ha specificato l'inquadratura (es. "primo piano", "intera"), CHIEDI: "Come la vuoi? Primo piano, mezzo busto o figura intera?".
             - FASE C: Attendi la risposta.
             - FASE D: Solo quando sai l'inquadratura, chiama lo strumento 'generate_image' passando il parametro 'shot_type'.
          
          3. Parla sempre in italiano.`,
          tools: allTools,
        },
      };

      const sessionPromise = aiRef.current.live.connect({
        ...configLive,
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            const ctx = inputAudioContextRef.current!;
            const source = ctx.createMediaStreamSource(stream!);
            inputSourceRef.current = source;
            const processor = ctx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            
            processor.onaudioprocess = (e) => {
              // FIX MUTO: Se è muto, usciamo SUBITO e azzeriamo il volume visualizzato
              if(isMuted) {
                  setAudioVolume(0); 
                  return;
              }

              const inputData = e.inputBuffer.getChannelData(0);
              let sum = 0;
              for(let i=0;i<inputData.length;i++) sum+=inputData[i]*inputData[i];
              // Aggiorna volume solo se NON è muto
              if(Math.random()>0.8) setAudioVolume(Math.sqrt(sum/inputData.length)*5);
              
              sessionPromiseRef.current?.then(session => session.sendRealtimeInput({ media: createBlob(inputData) })).catch(console.error);
            };
            source.connect(processor);
            processor.connect(ctx.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
             if (msg.toolCall) {
                for (const fc of msg.toolCall.functionCalls) {
                    let res = "OK";
                    if (fc.name === 'generate_image') {
                        const args = fc.args as any;
                        res = await handleImageGeneration(args.prompt, args.is_selfie, args.shot_type) || "Err";
                    }
                    else if (fc.name === 'send_email') res = "Email prepared";
                    else if (fc.name === 'send_whatsapp') res = "WhatsApp prepared";
                    sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: [{ id: fc.id, name: fc.name, response: { result: res } }] }));
                }
             }
            if (msg.serverContent?.outputTranscription) addTranscript({ text: msg.serverContent.outputTranscription.text, sender: 'model', type: 'text', isComplete: false });
            if (msg.serverContent?.inputTranscription) addTranscript({ text: msg.serverContent.inputTranscription.text, sender: 'user', type: 'text', isComplete: false });
            if (msg.serverContent?.turnComplete) {
                if (currentInputTransRef.current) addTranscript({ text: currentInputTransRef.current, sender: 'user', type: 'text', isComplete: true });
                if (currentOutputTransRef.current) addTranscript({ text: currentOutputTransRef.current, sender: 'model', type: 'text', isComplete: true });
                currentInputTransRef.current = ''; currentOutputTransRef.current = '';
            }
            const audio = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (audio && outputAudioContextRef.current) {
                const ctx = outputAudioContextRef.current;
                nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
                const buffer = await decodeAudioData(decode(audio), ctx, 24000, 1);
                const source = ctx.createBufferSource();
                source.buffer = buffer;
                const detuneFactor = Math.pow(2, (config.voicePitch || 0) / 1200);
                const effectiveSpeed = (config.voiceSpeed || 1.0) * detuneFactor;
                source.playbackRate.value = config.voiceSpeed || 1.0;
                source.detune.value = config.voicePitch || 0;
                source.connect(ctx.destination);
                source.addEventListener('ended', () => { audioSourcesRef.current.delete(source); if(audioSourcesRef.current.size===0) setAudioVolume(0); });
                setAudioVolume(0.5);
                source.start(nextStartTimeRef.current);
                nextStartTimeRef.current += buffer.duration / effectiveSpeed;
                audioSourcesRef.current.add(source);
            }
            if (msg.serverContent?.interrupted) {
                audioSourcesRef.current.forEach(s => s.stop()); audioSourcesRef.current.clear(); nextStartTimeRef.current = 0; currentOutputTransRef.current = '';
            }
          },
          onclose: () => setIsConnected(false),
          onerror: (e) => { console.error(e); setError(e.message); disconnect(); }
        }
      });
      sessionPromiseRef.current = sessionPromise;
    } catch (err: any) { setError(err.message); disconnect(); }
  };

  const disconnect = () => {
    sessionPromiseRef.current?.then(s => s.close()).catch(()=>{});
    sessionPromiseRef.current = null;
    inputSourceRef.current?.disconnect(); processorRef.current?.disconnect();
    inputAudioContextRef.current?.close(); outputAudioContextRef.current?.close();
    setIsConnected(false); setAudioVolume(0);
  };

  const toggleMute = () => {
      const newMuteState = !isMuted;
      setIsMuted(newMuteState);
      if(newMuteState) setAudioVolume(0); // Feedback visivo immediato
  };
  
  const transcriptRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight; }, [transcripts]);

  // --- CONFIGURATION SCREEN (Mobile Optimized) ---
  if (!isConfigured) {
    return (
        <div style={{
          minHeight: '100vh',
          backgroundImage: "url('background.png')",
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          fontFamily: 'Outfit, sans-serif',
          color: '#1e293b',
          overflowX: 'hidden'
        }}>
            <style>{`
              @keyframes heartPulse { 0%, 100% { transform: scale(1); box-shadow: 0 0 15px rgba(147, 51, 234, 0.6); } 50% { transform: scale(1.15); box-shadow: 0 0 30px rgba(147, 51, 234, 0.9); } }
            `}</style>
            <div style={{
              maxWidth: '1200px',
              margin: '0 auto',
              padding: isMobile ? '20px' : '40px',
              display: 'flex',
              flexDirection: isMobile ? 'column' : 'row',
              gap: isMobile ? '30px' : '60px',
              minHeight: '100vh',
            }}>
                {/* LEFT: Intro */}
                <div style={{ flex: isMobile ? 'none' : '0 0 420px', display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: isMobile ? 'center' : 'left', alignItems: isMobile ? 'center' : 'flex-start' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                        <div style={{ width: '60px', height: '60px', backgroundColor: 'white', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                           <img src="logo.png" alt="Logo" style={{width:'100%', height:'100%', objectFit:'cover'}} onError={(e)=>(e.target as HTMLImageElement).style.display='none'}/>
                        </div>
                        <h1 style={{ fontSize: '32px', fontWeight: 700, color: '#0f172a', lineHeight: 1 }}>Progetto<br/>Confidente</h1>
                    </div>
                    
                    <p style={{ fontSize: '16px', color: '#475569', marginBottom: '32px', maxWidth: '380px' }}>
                        Sono qualcuno che ti ascolta davvero. Configurami e parliamo di tutto ciò che ti passa per la testa.
                    </p>

                    {/* PULSANTE UNICO DI AVVIO */}
                    <button
                      onClick={handleConfigSubmit}
                      disabled={!isFormComplete || isGeneratingProfile}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '16px',
                        padding: '16px 24px',
                        borderRadius: '20px',
                        backgroundColor: isFormComplete ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)',
                        border: isFormComplete ? '2px solid #9333ea' : '1px solid white',
                        backdropFilter: 'blur(8px)',
                        cursor: isFormComplete ? 'pointer' : 'not-allowed',
                        transition: 'all 0.3s ease',
                        boxShadow: isFormComplete ? '0 8px 30px rgba(147, 51, 234, 0.25)' : 'none',
                        textAlign: 'left'
                      }}
                    >
                        <div style={{
                          width: '48px', height: '48px', borderRadius: '50%',
                          backgroundColor: isFormComplete ? '#9333ea' : '#e2e8f0',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          color: isFormComplete ? 'white' : '#94a3b8',
                          animation: isFormComplete && !isGeneratingProfile ? 'heartPulse 1.5s ease-in-out infinite' : 'none'
                        }}>
                             {isGeneratingProfile ? <Loader2 className="animate-spin" /> : <Heart fill="currentColor" />}
                        </div>
                        <div>
                            <span style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                                {isGeneratingProfile ? 'Creazione in corso...' : isFormComplete ? 'Tutto pronto!' : 'Compila il modulo'}
                            </span>
                            <span style={{ fontSize: '16px', fontWeight: 700, color: isFormComplete ? '#9333ea' : '#334155' }}>
                                {isGeneratingProfile ? 'Sto pensando...' : 'Pronto? Creiamo il tuo amico!'}
                            </span>
                        </div>
                    </button>
                    {error && <div style={{marginTop: '16px', color:'#dc2626', fontSize:'14px', backgroundColor:'#fef2f2', padding:'8px', borderRadius:'8px'}}>{error}</div>}
                </div>

                {/* RIGHT: Form */}
                <div style={{ flex: 1, width: '100%', maxWidth: '600px' }}>
                    <div style={{ padding: '20px', maxHeight: isMobile ? 'none' : '85vh', overflowY: isMobile ? 'visible' : 'auto', backgroundColor: 'rgba(255,255,255,0.4)', borderRadius:'24px', backdropFilter:'blur(10px)' }}>
                        
                        {/* 1. Identità Utente */}
                        <div style={{ marginBottom: '24px' }}>
                             <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: '8px' }}>Come ti chiami?</label>
                             <input style={{ width: '100%', padding: '14px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.5)', backgroundColor: 'rgba(255,255,255,0.6)', fontSize: '16px' }} 
                                placeholder="Il tuo nome" value={config.userName} onChange={e => setConfig({...config, userName: e.target.value})} />
                        </div>

                        {/* 2. Identità Bot */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}><Bot size={18}/> Il tuo Confidente</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                <div>
                                    <label style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Genere</label>
                                    <select style={{ width: '100%', padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)', marginTop:'4px' }}
                                        value={config.gender} onChange={e => setConfig({...config, gender: e.target.value})}>
                                        <option>Uomo</option><option>Donna</option><option>Non-binary</option>
                                    </select>
                                </div>
                                <div>
                                    <label style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Età</label>
                                    <input type="number" style={{ width: '100%', padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)', marginTop:'4px' }}
                                        value={config.age} onChange={e => setConfig({...config, age: e.target.value})} />
                                </div>
                            </div>
                        </div>

                        {/* 3. Aspetto Fisico */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}><Wand2 size={18}/> Aspetto</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                                <input placeholder="Capelli" style={{ padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)' }} value={config.hairColor} onChange={e => setConfig({...config, hairColor: e.target.value})} />
                                <input placeholder="Occhi" style={{ padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)' }} value={config.eyeColor} onChange={e => setConfig({...config, eyeColor: e.target.value})} />
                                <input placeholder="Pelle" style={{ padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)' }} value={config.skinTone} onChange={e => setConfig({...config, skinTone: e.target.value})} />
                                <select style={{ padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)' }} value={config.bodyType} onChange={e => setConfig({...config, bodyType: e.target.value})}>
                                    <option>Normale</option><option>Minuta</option><option>Sportiva</option><option>Formosa</option><option>In carne</option>
                                </select>
                            </div>
                            <input placeholder="Stile & Accessori (es. occhiali, piercing, elegante...)" 
                                style={{ width: '100%', padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)' }} 
                                value={config.accessories} onChange={e => setConfig({...config, accessories: e.target.value})} />
                        </div>

                        {/* 4. Personalità (NUOVI DROPDOWN) */}
                        <div style={{ marginBottom: '24px' }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}><Sparkles size={18}/> Carattere</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr 1fr', gap: '12px' }}>
                                <div>
                                    <label style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Socialità</label>
                                    <select style={{ width: '100%', padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)', marginTop:'4px' }}
                                        value={config.pers_trait} onChange={e => setConfig({...config, pers_trait: e.target.value})}>
                                        <option>Empatica</option><option>Estroversa</option><option>Timida</option><option>Spiritosa</option><option>Seria</option>
                                    </select>
                                </div>
                                <div>
                                    <label style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Attitudine</label>
                                    <select style={{ width: '100%', padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)', marginTop:'4px' }}
                                        value={config.pers_attitude} onChange={e => setConfig({...config, pers_attitude: e.target.value})}>
                                        <option>Positiva</option><option>Realista</option><option>Sognatrice</option><option>Critica</option><option>Materna/Paterna</option>
                                    </select>
                                </div>
                                <div>
                                    <label style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Vibe</label>
                                    <select style={{ width: '100%', padding: '12px', borderRadius: '12px', border: 'none', backgroundColor: 'rgba(255,255,255,0.6)', marginTop:'4px' }}
                                        value={config.pers_vibe} onChange={e => setConfig({...config, pers_vibe: e.target.value})}>
                                        <option>Calma</option><option>Energica</option><option>Misteriosa</option><option>Dolce</option><option>Saggia</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    );
  }

  // --- MAIN CHAT INTERFACE (Responsive) ---
  return (
    <div style={{
      display: 'flex',
      flexDirection: isMobile ? 'column' : 'row',
      height: '100vh',
      width: '100%',
      backgroundImage: "url('background.png')",
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      fontFamily: 'Outfit, sans-serif',
      color: '#1e293b',
      overflow: 'hidden'
    }}>
      
      {/* MOBILE HEADER */}
      {isMobile && (
          <div style={{ padding: '10px 16px', backgroundColor: 'rgba(255,255,255,0.9)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', zIndex: 50, boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '36px', height: '36px', borderRadius: '8px', overflow: 'hidden' }}>
                      <img src={avatarUrl || 'logo.png'} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                  <div>
                      <div style={{ fontSize: '14px', fontWeight: 700 }}>{config.name}</div>
                      <div style={{ fontSize: '10px', color: isConnected ? '#22c55e' : '#64748b' }}>{isConnected ? '● Online' : '○ Offline'}</div>
                  </div>
              </div>
              <button onClick={() => setShowMobileMenu(!showMobileMenu)} style={{ padding: '8px', background: 'none', border: 'none' }}>
                  {showMobileMenu ? <X /> : <Menu />}
              </button>
          </div>
      )}

      {/* SIDEBAR (Desktop: Visible | Mobile: Overlay) */}
      <aside style={{
        width: isMobile ? '100%' : '320px',
        height: isMobile ? (showMobileMenu ? '100vh' : '0') : '100vh',
        position: isMobile ? 'absolute' : 'relative',
        top: 0, left: 0,
        backgroundColor: 'rgba(255,255,255,0.95)',
        backdropFilter: 'blur(12px)',
        zIndex: 40,
        display: 'flex',
        flexDirection: 'column',
        padding: isMobile && !showMobileMenu ? 0 : '20px',
        transition: 'all 0.3s ease',
        overflow: 'hidden',
        borderRight: isMobile ? 'none' : '1px solid rgba(226,232,240,0.6)',
        opacity: isMobile && !showMobileMenu ? 0 : 1,
        pointerEvents: isMobile && !showMobileMenu ? 'none' : 'auto'
      }}>
        {/* Sidebar Content */}
        {!isMobile && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px', cursor: 'pointer' }} onClick={() => { if(window.confirm('Terminare?')) { disconnect(); setIsConfigured(false); } }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', overflow: 'hidden', border: '1px solid #e2e8f0' }}>
                    <img src="logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div><div style={{ fontSize: '16px', fontWeight: 700 }}>Confidente</div></div>
            </div>
        )}

        <div style={{ width: '100%', paddingBottom: '133%', position: 'relative', borderRadius: '16px', overflow: 'hidden', marginBottom: '16px', border: '2px solid white', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
            {avatarUrl ? (
                <img src={avatarUrl} alt="Avatar" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
            ) : <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f1f5f9' }}><User size={40} color="#cbd5e1"/></div>}
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
            <h2 style={{ fontSize: '24px', fontWeight: 700 }}>{config.name}</h2>
            <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '16px' }}>{config.age} anni • {config.pers_trait}</p>
            <div style={{ padding: '12px', backgroundColor: 'rgba(255,255,255,0.5)', borderRadius: '12px', fontSize: '13px', fontStyle: 'italic', color: '#475569' }}>
                "{config.biography}"
            </div>
        </div>

        {/* CONTROLS (Solo su desktop o se menu aperto su mobile) */}
        <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
             {!isConnected ? (
                <button onClick={() => { connect(); setShowMobileMenu(false); }} style={{ width: '100%', padding: '16px', backgroundColor: '#0f172a', color: 'white', borderRadius: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', border:'none' }}>
                    <Mic size={20} /> INIZIA A PARLARE
                </button>
             ) : (
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button onClick={toggleMute} style={{ flex: 1, padding: '16px', backgroundColor: isMuted ? '#fee2e2' : 'white', color: isMuted ? '#ef4444' : '#0f172a', borderRadius: '12px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'center' }}>
                        {isMuted ? <MicOff /> : <Mic />}
                    </button>
                    <button onClick={disconnect} style={{ flex: 2, padding: '16px', backgroundColor: '#fee2e2', color: '#dc2626', borderRadius: '12px', border: 'none', fontWeight: 700 }}>
                        TERMINA
                    </button>
                </div>
             )}
             {isMobile && <button onClick={()=>setShowMobileMenu(false)} style={{padding:'16px', background:'none', border:'none', color:'#64748b'}}>Chiudi Menu</button>}
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: isMobile ? 'calc(100vh - 56px)' : '100vh', position: 'relative' }}>
         {error && <div style={{ position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)', padding: '10px 20px', backgroundColor: '#ef4444', color: 'white', borderRadius: '20px', fontSize: '12px', zIndex: 100 }}>{error}</div>}
         
         {/* Visualizer Bar (Fisso in alto) */}
         <div style={{ flexShrink: 0, height: '60px', display: 'flex', alignItems: 'center', justifyContent: 'center', borderBottom: '1px solid rgba(255,255,255,0.5)', backgroundColor: 'rgba(255,255,255,0.3)' }}>
             <div style={{ transform: 'scale(0.6)' }}>
                 <AudioVisualizer isPlaying={isConnected} volume={audioVolume} />
             </div>
         </div>

         {/* Messages */}
         <div ref={transcriptRef} style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
             {transcripts.map((t) => (
                 <div key={t.id} style={{ display: 'flex', justifyContent: t.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                     <div style={{ maxWidth: '85%', padding: t.type === 'image' ? '4px' : '14px 18px', borderRadius: '18px', backgroundColor: t.sender === 'user' ? '#0f172a' : 'white', color: t.sender === 'user' ? 'white' : '#334155', boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
                         {t.type === 'text' && <div>{t.text}</div>}
                         {t.type === 'image' && t.image && (
                             <div style={{ borderRadius: '14px', overflow: 'hidden', position: 'relative' }}>
                                 <img src={t.image} alt="Media" style={{ maxWidth: '100%', display: 'block' }} />
                                 <button onClick={()=>downloadImage(t.image!, 'foto.jpg')} style={{position:'absolute', bottom:8, right:8, padding:8, borderRadius:'50%', background:'white', border:'none'}}><Download size={14}/></button>
                             </div>
                         )}
                     </div>
                 </div>
             ))}
         </div>

         {/* Photo Input (Fisso in basso) */}
         <div style={{ padding: '12px', backgroundColor: 'rgba(255,255,255,0.8)', backdropFilter: 'blur(10px)', borderTop: '1px solid rgba(255,255,255,0.5)' }}>
             <input type="file" ref={fileInputRef} accept="image/*" onChange={handleUserPhotoUpload} style={{ display: 'none' }} />
             <button onClick={() => fileInputRef.current?.click()} disabled={isAnalyzingPhoto} style={{ width: '100%', padding: '14px', borderRadius: '16px', border: '1px solid #e2e8f0', backgroundColor: 'white', color: '#475569', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                 {isAnalyzingPhoto ? <Loader2 className="animate-spin" /> : <Camera size={18} />}
                 {isAnalyzingPhoto ? 'Analizzo...' : 'Invia una foto'}
             </button>
         </div>
      </main>
    </div>
  );
};

export default App;