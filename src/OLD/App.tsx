import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type, Tool } from '@google/genai';
import { TranscriptItem, AssistantConfig } from './types';
import { createBlob, decode, decodeAudioData } from './utils/audio';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Mic, MicOff, PhoneOff, User, Bot, Sparkles, Image as ImageIcon, ArrowRight, Loader2, Heart, Info, Mail, MessageCircle, ExternalLink, Download, Wand2, UserCircle, Sliders, Music2, Menu, Camera } from 'lucide-react';

const LIVE_MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const IMAGE_MODEL_NAME = 'imagen-4.0-generate-001	';
const TEXT_MODEL_NAME = 'gemini-2.0-flash';

// --- TOOLS DEFINITION ---
const generateImageTool: FunctionDeclaration = {
  name: 'generate_image',
  description: 'Genera un\'immagine. Usalo quando l\'utente chiede di vedere qualcosa o chiede una TUA foto.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: { type: Type.STRING, description: 'La descrizione del contesto o della scena.' },
      is_selfie: { type: Type.BOOLEAN, description: 'TRUE se è una foto dell\'assistente, FALSE se oggetto generico.' }
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

// --- BRANDING COMPONENT (Updated to match Confidente style) ---
const AppLogo = ({ size = 48, className = "" }: { size?: number, className?: string }) => {
  const [imgError, setImgError] = useState(false);

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <div className="relative h-full w-full bg-white/90 border border-white/50 rounded-[1rem] flex items-center justify-center overflow-hidden shadow-lg shadow-purple-200/40 backdrop-blur-sm">
        {!imgError ? (
           <img 
             src="logo.png" 
             alt="Logo Confidente" 
             className="w-full h-full object-cover p-1"
             onError={() => setImgError(true)}
           />
        ) : (
          <div className="relative z-10 flex items-center justify-center w-full h-full">
            <Heart size={size * 0.45} className="text-purple-400 fill-purple-400 absolute left-[15%] top-[25%] opacity-95" />
            <Heart size={size * 0.45} className="text-amber-400 fill-amber-400 absolute right-[15%] top-[35%] opacity-95" />
          </div>
        )}
      </div>
    </div>
  );
};

const App: React.FC = () => {
  // Configuration State
  const [config, setConfig] = useState<AssistantConfig>({
    userName: '',
    gender: 'Donna',
    age: '25',
    hairColor: 'Castani',
    eyeColor: 'Verdi',
    skinTone: 'Chiara',
    bodyType: 'Normale',
    physicalTraits: 'Sorriso gentile',
    personality: 'Empatica, calma, saggia, buona ascoltatrice',
    name: '',
    biography: '',
    visualPrompt: '',
    voicePitch: 0,
    voiceSpeed: 1.0,
    voiceEnergy: 50,
    voiceTone: 50,
  });
  const [isConfigured, setIsConfigured] = useState(false);
  const [isGeneratingProfile, setIsGeneratingProfile] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  
  // Check if all required fields are filled for the pulsing heart
  const isFormComplete = config.userName.trim() !== '' && 
                         config.gender !== '' && 
                         config.age !== '' && 
                         config.hairColor !== '' && 
                         config.eyeColor !== '' && 
                         config.skinTone !== '' &&
                         config.bodyType !== '' &&
                         config.personality !== '';

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
    // VERCEL FIX: Defensive API Key retrieval
    // import.meta.env might be undefined in some build contexts, causing the crash.
    let apiKey = '';
    try {
        // Use optional chaining to safely access VITE_API_KEY
        // @ts-ignore
        apiKey = import.meta.env?.VITE_API_KEY;
    } catch(e) {
        console.warn("import.meta.env access error", e);
    }

    // Fallback to process.env if available (sometimes polyfilled)
    if (!apiKey) {
        try {
            // @ts-ignore
            apiKey = process.env?.VITE_API_KEY || process.env?.API_KEY;
        } catch(e) {}
    }

    if (apiKey) {
      aiRef.current = new GoogleGenAI({ apiKey });
    } else {
      console.warn("API Key mancante. Assicurati che VITE_API_KEY sia impostata nelle variabili d'ambiente di Vercel.");
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

  // Funzione per gestire l'upload di foto da parte dell'utente
  const handleUserPhotoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !aiRef.current) return;

    setIsAnalyzingPhoto(true);
    
    try {
      // Converti l'immagine in base64
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Data = e.target?.result as string;
        
        // Aggiungi l'immagine dell'utente alla chat
        const userImageId = Date.now().toString();
        addTranscript({
          id: userImageId,
          sender: 'user',
          type: 'image',
          image: base64Data,
          isComplete: true
        });

        try {
          // Chiedi all'IA di commentare l'immagine
          const imageAnalysisPrompt = `Sei ${config.name}, un amico empatico e curioso. L'utente ${config.userName} ti ha appena inviato una foto. 
          Analizza l'immagine e rispondi in modo amichevole e caloroso. 
          Fai commenti positivi su quello che vedi, mostra interesse genuino e fai 1-2 domande per stimolare la conversazione.
          Sii naturale e colloquiale, come un vero amico. Rispondi in italiano, max 2-3 frasi.`;

          const response = await aiRef.current!.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: [
              {
                role: 'user',
                parts: [
                  { text: imageAnalysisPrompt },
                  { 
                    inlineData: {
                      mimeType: file.type,
                      data: base64Data.split(',')[1]
                    }
                  }
                ]
              }
            ]
          });

          const aiComment = response.text || "Che bella foto! Raccontami di più!";
          
          // Aggiungi alla chat
          addTranscript({
            id: (Date.now() + 1).toString(),
            sender: 'model',
            type: 'text',
            text: aiComment,
            isComplete: true
          });
          
          // Se la sessione live è attiva, invia il testo per farlo pronunciare dall'IA
          if (isConnected && sessionPromiseRef.current) {
            try {
              const session = await sessionPromiseRef.current;
              // Invia un messaggio di testo alla sessione per farlo leggere con la voce dell'assistente
              session.sendClientContent({ turns: [{ role: 'user', parts: [{ text: `Leggi ad alta voce questo commento che hai appena scritto sulla foto: "${aiComment}"` }] }] });
            } catch (e) {
              console.log('Sessione non disponibile per TTS');
            }
          }

        } catch (err) {
          console.error('Errore analisi foto:', err);
          const fallbackText = "Che bella foto! Mi piacerebbe saperne di più. Cosa stavi facendo in quel momento?";
          addTranscript({
            id: (Date.now() + 1).toString(),
            sender: 'model',
            type: 'text',
            text: fallbackText,
            isComplete: true
          });
        }
        
        setIsAnalyzingPhoto(false);
      };
      
      reader.readAsDataURL(file);
    } catch (err) {
      console.error('Errore upload foto:', err);
      setIsAnalyzingPhoto(false);
    }
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleConfigSubmit = async () => {
    if (!aiRef.current) {
        setError("API Key mancante. Configura VITE_API_KEY su Vercel.");
        return;
    }
    setIsGeneratingProfile(true);
    setError(null);

    try {
        const hasManualName = config.name && config.name.trim().length > 0;
        setLoadingStep(hasManualName ? `Sto definendo la personalità di ${config.name}...` : 'Sto creando il tuo  amico ideale...');
        
        const basePrompt = `Crea un profilo per un COMPAGNO UMANO: Genere ${config.gender}, Età ${config.age}, Capelli ${config.hairColor}, Occhi ${config.eyeColor}, Pelle ${config.skinTone}, Corporatura ${config.bodyType || 'Normale'}, Caratteristiche fisiche: ${config.physicalTraits}, Personalità ${config.personality}.`;
        const nameInstruction = hasManualName ? `Il nome è "${config.name}".` : `Inventa un nome italiano creativo.`;

        const profilePrompt = `${basePrompt} ${nameInstruction} Rispondi JSON: {name, biography, visualPrompt}. La biography deve includere hobby, studi, esperienze. Il visualPrompt deve essere dettagliato per generare un ritratto fotorealistico.`;
        
        const textResponse = await aiRef.current.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: profilePrompt,
            config: { responseMimeType: 'application/json', responseSchema: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, biography: { type: Type.STRING }, visualPrompt: { type: Type.STRING } }, required: ['name', 'biography', 'visualPrompt'] } }
        });

        const profileData = JSON.parse(textResponse.text || '{}');
        if (!profileData.name) throw new Error("Errore generazione profilo.");

        setConfig(prev => ({ ...prev, name: profileData.name, biography: profileData.biography, visualPrompt: profileData.visualPrompt }));
        setLoadingStep(`Sto scattando una foto a ${profileData.name}...`);
        
        let foundUrl: string | null = null;
        
        try {
            // Cerca questa riga e sostituiscila:
// Usiamo "American shot" (piano americano) o "3/4 shot" per forzare l'inquadratura fino alle anche/pancia.
            // Aggiungiamo "hands visible" (mani visibili) perché aiuta l'IA a capire che deve inquadrare anche il corpo.
            const imagePrompt = `Medium shot from hips up (American shot), visible waist and stomach, camera distance 3 meters. The subject is a friendly ${config.gender === 'Donna' ? 'woman' : config.gender === 'Uomo' ? 'man' : 'person'}, ${config.age} years old, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${config.bodyType || 'normal'} build. Wearing casual-elegant clothes suitable for a full torso shot. 8k resolution, photorealistic, soft studio lighting. ${profileData.visualPrompt}`;            
            console.log('Generating image with imagen-4.0-generate-001:', imagePrompt);
            
            const imageResponse = await aiRef.current.models.generateImages({
                model: IMAGE_MODEL_NAME,
                prompt: imagePrompt,
                config: {
                    numberOfImages: 1,
                    outputMimeType: 'image/jpeg',
                    aspectRatio: '3:4'
                }
            });

            console.log('Risposta Imagen:', imageResponse);

            if (imageResponse.generatedImages && imageResponse.generatedImages.length > 0) {
                const img = imageResponse.generatedImages[0];
                if (img.image?.imageBytes) {
                    foundUrl = `data:image/jpeg;base64,${img.image.imageBytes}`;
                    console.log('Avatar generato con successo!');
                }
            }
            
            if (!foundUrl) {
                console.warn('Nessuna immagine nella risposta Imagen');
            }
        } catch (imgError: any) {
            console.error('Errore generazione immagine avatar:', imgError.message || imgError);
            // Continua senza avatar
        }
        
        setAvatarUrl(foundUrl);
        setIsConfigured(true);
    } catch (e: any) {
        setError("Errore creazione: " + e.message);
    } finally {
        setIsGeneratingProfile(false);
        setLoadingStep('');
    }
  };

  const handleImageGeneration = async (prompt: string, isSelfie: boolean = false): Promise<string | null> => {
    if (!aiRef.current) return null;
    
    // 1. Lanciamo la promessa di generazione SUBITO (in background)
    // Così l'immagine si carica mentre l'IA parla, guadagnando tempo.
    let finalPrompt = prompt;
    if (isSelfie && config.visualPrompt) {
        finalPrompt = `Professional Medium shot, waist-up photograph of a friendly ${config.gender === 'Donna' ? 'woman' : config.gender === 'Uomo' ? 'man' : 'person'}, ${config.age} years old, ${config.hairColor} hair, ${config.eyeColor} eyes. ${config.visualPrompt}. ${prompt}. Photorealistic, warm smile, natural pose.`;
    } else {
        // Anche per le immagini non-selfie, se descrivono il personaggio, usiamo medium shot
        finalPrompt = `Medium shot, cinematic photo. ${prompt}`;
    }

    const imageGenerationPromise = aiRef.current.models.generateImages({
        model: IMAGE_MODEL_NAME,
        prompt: finalPrompt,
        config: {
            numberOfImages: 1,
            outputMimeType: 'image/jpeg',
            // MODIFICA FORMATO:
            // '4:3' = Orizzontale (Landscape) -> Soddisfa "non in formato ritratto"
            // '3:4' = Verticale (Formato telefono)
            // '1:1' = Quadrato
            aspectRatio: '3:4'
        }
    });

    try {
        if (isSelfie) {
            // 2. SIMULAZIONE TEMPO SCENICO
            // Aggiungiamo un messaggio visivo immediato
            addTranscript({ sender: 'model', type: 'text', text: `📸 *Prende il telefono e si mette in posa...*`, isComplete: true });
            
            // 3. IL TRUCCO: Aspettiamo 4 secondi TASSATIVI mentre l'audio (la voce) sta andando.
            // Questo permette all'IA di dire "Ehm... aspetta... click!" PRIMA che appaia la foto.
            await new Promise(resolve => setTimeout(resolve, 4000));
        } else {
            addTranscript({ sender: 'model', type: 'text', text: `🎨 Genero l'immagine: "${prompt}"`, isComplete: true });
        }

        // 4. Ora aspettiamo che l'immagine sia pronta (se ci ha messo meno di 4s, è già pronta)
        const response = await imageGenerationPromise;

        let imageUrl: string | null = null;
        if (response.generatedImages && response.generatedImages.length > 0) {
            const img = response.generatedImages[0];
            if (img.image?.imageBytes) {
                imageUrl = `data:image/jpeg;base64,${img.image.imageBytes}`;
            }
        }
        
        if (imageUrl) {
            // 5. SOLO ORA mostriamo l'immagine
            addTranscript({ sender: 'model', type: 'image', image: imageUrl, isComplete: true });
            return isSelfie ? "Foto inviata!" : "Ecco l'immagine.";
        }
        return "Errore nella generazione.";
    } catch (e: any) {
        console.error('Errore generazione immagine:', e.message || e);
        return "Mi sa che la cazzo di fotocamera non funziona bene oggi...";
    }
  };

  const handleSendEmail = (recipient: string, subject: string, body: string) => {
    addTranscript({ sender: 'model', type: 'action', text: `📧 Email pronta per: ${recipient}`, isComplete: true, actionUrl: `mailto:${recipient}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`, actionLabel: 'Invia Email', actionIcon: 'mail' });
    return "SUCCESS";
  };

  const handleSendWhatsapp = (phoneNumber: string, text: string) => {
    addTranscript({ sender: 'model', type: 'action', text: `💬 WhatsApp pronto per: ${phoneNumber}`, isComplete: true, actionUrl: `https://wa.me/${phoneNumber.replace(/\D/g, '')}?text=${encodeURIComponent(text)}`, actionLabel: 'Invia WhatsApp', actionIcon: 'message-circle' });
    return "SUCCESS";
  };

  const connect = async () => {
    if (!aiRef.current) {
        setError("Chiave API non trovata. Controlla le impostazioni di Vercel.");
        return;
    }
    setError(null);
    try {
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const ageNum = parseInt(config.age) || 30;
      let selectedVoiceName = config.gender === 'Uomo' ? (ageNum < 35 ? 'Puck' : 'Fenrir') : (ageNum < 35 ? 'Aoede' : 'Kore');

      const configLive = {
        model: LIVE_MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoiceName } } },
          systemInstruction: `Sei ${config.name}, amico/a di ${config.userName}. Personalità: ${config.personality}. Bio: ${config.biography}. 
          
REGOLE FONDAMENTALI:
- Sii naturale, empatico, colloquiale. Non dire MAI di essere un'AI o un assistente virtuale.
- Quando ${config.userName} ti chiede una TUA foto o un selfie, NON chiamare subito lo strumento. PRIMA devi parlare:reagisci con un po' di IMBARAZZO e timidezza, come farebbe una persona vera. Esempio: "Oh... una mia foto? Ehm, non so se vengo bene in foto..." poi dopo un attimo di esitazione accetta: "Va bene dai, aspetta che me la scatto... eccola!"
- CHIAMA lo strumento 'generate_image' SOLO ALLA FINE della tua frase parlata.
- L'ordine deve essere: [PARLATO DI IMBARAZZO] -> [CHIAMATA STRUMENTO].
- Quando ${config.userName} ti invia una foto sua, commentala con entusiasmo e curiosità genuina, fai domande per saperne di più.
- Parla sempre in italiano in modo naturale e amichevole.`,
          tools: allTools,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
      };

      const sessionPromise = aiRef.current.live.connect({
        ...configLive,
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            const ctx = inputAudioContextRef.current!;
            const source = ctx.createMediaStreamSource(stream);
            inputSourceRef.current = source;
            const processor = ctx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              let sum = 0;
              for(let i=0;i<inputData.length;i++) sum+=inputData[i]*inputData[i];
              if(Math.random()>0.8) setAudioVolume(Math.sqrt(sum/inputData.length)*5);
              if(isMuted) return;
              sessionPromiseRef.current?.then(session => session.sendRealtimeInput({ media: createBlob(inputData) })).catch(console.error);
            };
            source.connect(processor);
            processor.connect(ctx.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
             if (msg.toolCall) {
                for (const fc of msg.toolCall.functionCalls) {
                    let res = "OK";
                    if (fc.name === 'generate_image') res = await handleImageGeneration((fc.args as any).prompt, (fc.args as any).is_selfie) || "Err";
                    else if (fc.name === 'send_email') res = handleSendEmail((fc.args as any).recipient, (fc.args as any).subject, (fc.args as any).body);
                    else if (fc.name === 'send_whatsapp') res = handleSendWhatsapp((fc.args as any).phoneNumber, (fc.args as any).text);
                    sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: [{ id: fc.id, name: fc.name, response: { result: res } }] }));
                }
             }
            if (msg.serverContent?.outputTranscription) {
                currentOutputTransRef.current += msg.serverContent.outputTranscription.text;
                addTranscript({ text: currentOutputTransRef.current, sender: 'model', type: 'text', isComplete: false });
            }
            if (msg.serverContent?.inputTranscription) {
                currentInputTransRef.current += msg.serverContent.inputTranscription.text;
                addTranscript({ text: currentInputTransRef.current, sender: 'user', type: 'text', isComplete: false });
            }
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
                source.playbackRate.value = config.voiceSpeed || 1.0;
                source.detune.value = config.voicePitch || 0;
                
                // CRITICAL FIX: Calculate effective speed to prevent crackling/gaps
                // Detune affects speed (100 cents = 1 semitone = approx 5.9% speed change)
                // Formula: speed * 2^(cents/1200)
                const detuneFactor = Math.pow(2, (config.voicePitch || 0) / 1200);
                const effectiveSpeed = (config.voiceSpeed || 1.0) * detuneFactor;

                source.connect(ctx.destination);
                source.addEventListener('ended', () => { audioSourcesRef.current.delete(source); if(audioSourcesRef.current.size===0) setAudioVolume(0); });
                setAudioVolume(0.5);
                source.start(nextStartTimeRef.current);
                // Advance the cursor by the ACTUAL duration played (original duration / effective speed)
                nextStartTimeRef.current += buffer.duration / effectiveSpeed;
                audioSourcesRef.current.add(source);
            }
            if (msg.serverContent?.interrupted) {
                audioSourcesRef.current.forEach(s => s.stop()); audioSourcesRef.current.clear(); nextStartTimeRef.current = 0; currentOutputTransRef.current = '';
            }
          },
          onclose: () => setIsConnected(false),
          onerror: (e) => { console.error(e); disconnect(); }
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

  const toggleMute = () => setIsMuted(!isMuted);
  const transcriptRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight; }, [transcripts]);

  // --- CONFIGURATION SCREEN (LIGHT THEME WATERCOLOR STYLE) ---
  if (!isConfigured) {
    return (
        <div style={{
          minHeight: '100vh',
          backgroundImage: "url('background.png')",
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          position: 'relative',
          fontFamily: 'Outfit, sans-serif',
          color: '#1e293b'
        }}>
            {/* CSS Animation for pulsing heart */}
            <style>{`
              @keyframes heartPulse {
                0%, 100% { transform: scale(1); box-shadow: 0 0 15px rgba(147, 51, 234, 0.6); }
                50% { transform: scale(1.15); box-shadow: 0 0 30px rgba(147, 51, 234, 0.9); }
              }
              input[type="range"]::-webkit-slider-thumb {
                appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #9333ea;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(147, 51, 234, 0.4);
              }
            `}</style>
            {/* Main Container - Two Columns */}
            <div style={{
              maxWidth: '1200px',
              margin: '0 auto',
              padding: '40px',
              position: 'relative',
              zIndex: 10,
              display: 'flex',
              flexDirection: 'row',
              gap: '60px',
              minHeight: '100vh',
              alignItems: 'center'
            }}>
                
                {/* LEFT COLUMN: Brand & Description */}
                <div style={{ flex: '0 0 420px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    
                    {/* Logo + Project Name - LOGO PIÙ GRANDE */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
                        <div style={{
                          width: '80px',
                          height: '80px',
                          backgroundColor: 'rgba(255,255,255,0.95)',
                          borderRadius: '16px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          boxShadow: '0 4px 24px rgba(147, 112, 219, 0.25)',
                          overflow: 'hidden'
                        }}>
                          <img src="logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                        </div>
                        <div>
                            <div style={{ fontSize: '12px', fontWeight: 700, letterSpacing: '0.2em', color: '#64748b', textTransform: 'uppercase' }}>Progetto</div>
                            <div style={{ fontSize: '22px', fontWeight: 700, color: '#1e293b', letterSpacing: '-0.02em' }}>CONFIDENTE</div>
                        </div>
                    </div>

                    {/* Main Title */}
                    <h1 style={{
                      fontSize: '56px',
                      fontWeight: 700,
                      color: '#0f172a',
                      marginBottom: '24px',
                      lineHeight: 1.1,
                      letterSpacing: '-0.02em'
                    }}>
                        Amico<br/>Confidente
                    </h1>
                    
                    {/* Description */}
                    <p style={{
                      fontSize: '17px',
                      color: '#475569',
                      fontWeight: 500,
                      lineHeight: 1.7,
                      maxWidth: '380px',
                      marginBottom: '32px'
                    }}>
                        Sono qualcuno che ti ascolta davvero. 
                        Configurami, dammi un volto e una voce, e parliamo di tutto ciò che ti passa per la testa.
                    </p>

                    {/* Feature Badge - Cuore lampeggiante quando form è completo */}
                    <div style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '16px',
                      padding: '16px 24px 16px 16px',
                      borderRadius: '16px',
                      backgroundColor: isFormComplete ? 'rgba(147, 51, 234, 0.1)' : 'rgba(255,255,255,0.7)',
                      border: isFormComplete ? '2px solid rgba(147, 51, 234, 0.5)' : '1px solid rgba(255,255,255,0.8)',
                      backdropFilter: 'blur(8px)',
                      width: 'fit-content',
                      transition: 'all 0.3s ease',
                      boxShadow: isFormComplete ? '0 0 20px rgba(147, 51, 234, 0.3)' : 'none'
                    }}>
                        <div style={{
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          backgroundColor: isFormComplete ? '#9333ea' : '#f3e8ff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: isFormComplete ? 'white' : '#9333ea',
                          animation: isFormComplete ? 'heartPulse 1s ease-in-out infinite' : 'none',
                          boxShadow: isFormComplete ? '0 0 15px rgba(147, 51, 234, 0.6)' : 'none'
                        }}>
                            <Heart fill="currentColor" size={20} />
                        </div>
                        <span style={{ fontWeight: 600, color: isFormComplete ? '#9333ea' : '#334155' }}>
                          {isFormComplete ? 'Pronto! Creiamo il tuo amico!' : 'Ascolto Attivo'}
                        </span>
                    </div>
                    
                    {/* Spacer */}
                    <div style={{ flex: 1 }} />
                    
                    {/* Copyright */}
                    <div style={{
                      marginTop: '40px',
                      fontSize: '10px',
                      fontWeight: 500,
                      color: '#94a3b8',
                      letterSpacing: '0.05em'
                    }}>
                        © Copyright Effetre Properties IA Division 2025 - All rights reserved
                    </div>
                </div>

                {/* RIGHT COLUMN: Configuration Form - TRASPARENTE */}
                <div style={{ flex: 1, maxWidth: '650px' }}>
                    {/* Form SENZA box bianco - completamente trasparente */}
                    <div style={{
                      padding: '20px',
                      maxHeight: '85vh',
                      overflowY: 'auto'
                    }}>
                        
                        {error && (
                            <div style={{ marginBottom: '24px', padding: '16px', backgroundColor: 'rgba(254,242,242,0.9)', border: '1px solid #fecaca', borderRadius: '12px', color: '#dc2626', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Info size={16} /> {error}
                            </div>
                        )}

                        {/* Section 1: Chi sei tu? */}
                        <div style={{ marginBottom: '28px' }}>
                            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <User size={20} style={{ color: '#94a3b8' }}/> Chi sei tu?
                            </h3>
                            <div>
                                <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Il tuo Nome</label>
                                <input 
                                    style={{
                                      width: '100%',
                                      backgroundColor: 'rgba(255,255,255,0.75)',
                                      border: '1px solid rgba(226,232,240,0.6)',
                                      borderRadius: '16px',
                                      padding: '16px 20px',
                                      fontSize: '15px',
                                      color: '#1e293b',
                                      outline: 'none',
                                      fontWeight: 500,
                                      boxSizing: 'border-box',
                                      backdropFilter: 'blur(4px)'
                                    }}
                                    placeholder="Come vuoi che ti chiami?"
                                    value={config.userName}
                                    onChange={(e) => setConfig({...config, userName: e.target.value})}
                                />
                            </div>
                        </div>

                        {/* Section 2: Il tuo Confidente */}
                        <div style={{ marginBottom: '28px' }}>
                            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <Bot size={20} style={{ color: '#f59e0b' }}/> Il tuo Confidente
                            </h3>
                            
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                                <div>
                                    <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Genere Assistente</label>
                                    <div style={{ position: 'relative' }}>
                                        <select 
                                            style={{
                                              width: '100%',
                                              backgroundColor: 'rgba(255,255,255,0.75)',
                                              border: '1px solid rgba(226,232,240,0.6)',
                                              borderRadius: '16px',
                                              padding: '16px 20px',
                                              fontSize: '15px',
                                              color: '#1e293b',
                                              outline: 'none',
                                              cursor: 'pointer',
                                              fontWeight: 500,
                                              appearance: 'none',
                                              boxSizing: 'border-box',
                                              backdropFilter: 'blur(4px)'
                                            }}
                                            value={config.gender}
                                            onChange={(e) => setConfig({...config, gender: e.target.value})}
                                        >
                                            <option>Uomo</option>
                                            <option>Donna</option>
                                            <option>Non-binary</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '20px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: '#94a3b8' }}>▼</div>
                                    </div>
                                </div>
                                <div>
                                    <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Età Apparente</label>
                                    <input 
                                        type="number"
                                        style={{
                                          width: '100%',
                                          backgroundColor: 'rgba(255,255,255,0.75)',
                                          border: '1px solid rgba(226,232,240,0.6)',
                                          borderRadius: '16px',
                                          padding: '16px 20px',
                                          fontSize: '15px',
                                          color: '#1e293b',
                                          outline: 'none',
                                          fontWeight: 500,
                                          boxSizing: 'border-box',
                                          backdropFilter: 'blur(4px)'
                                        }}
                                        value={config.age}
                                        onChange={(e) => setConfig({...config, age: e.target.value})}
                                    />
                                </div>
                            </div>
                            
                            <div>
                                <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Nome Assistente (Opzionale)</label>
                                <input 
                                    style={{
                                      width: '100%',
                                      backgroundColor: 'rgba(255,255,255,0.75)',
                                      border: '1px solid rgba(226,232,240,0.6)',
                                      borderRadius: '16px',
                                      padding: '16px 20px',
                                      fontSize: '15px',
                                      color: '#1e293b',
                                      outline: 'none',
                                      fontWeight: 500,
                                      boxSizing: 'border-box',
                                      backdropFilter: 'blur(4px)'
                                    }}
                                    placeholder="Lascia vuoto per generare automaticamente"
                                    value={config.name}
                                    onChange={(e) => setConfig({...config, name: e.target.value})}
                                />
                            </div>
                        </div>

                        {/* Section 3: Dettagli Aspetto */}
                        <div style={{ marginBottom: '28px' }}>
                            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <Wand2 size={20} style={{ color: '#94a3b8' }}/> Aspetto Fisico
                            </h3>
                            
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                                <div>
                                    <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Capelli</label>
                                    <input 
                                      style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '12px', padding: '12px 16px', fontSize: '14px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)' }}
                                      value={config.hairColor} 
                                      onChange={(e) => setConfig({...config, hairColor: e.target.value})} 
                                    />
                                </div>
                                <div>
                                    <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Occhi</label>
                                    <input 
                                      style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '12px', padding: '12px 16px', fontSize: '14px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)' }}
                                      value={config.eyeColor} 
                                      onChange={(e) => setConfig({...config, eyeColor: e.target.value})} 
                                    />
                                </div>
                                <div>
                                    <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Pelle</label>
                                    <input 
                                      style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '12px', padding: '12px 16px', fontSize: '14px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)' }}
                                      value={config.skinTone} 
                                      onChange={(e) => setConfig({...config, skinTone: e.target.value})} 
                                    />
                                </div>
                                <div>
                                    <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Corporatura</label>
                                    <div style={{ position: 'relative' }}>
                                        <select 
                                            style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '12px', padding: '12px 16px', fontSize: '14px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)', appearance: 'none', cursor: 'pointer' }}
                                            value={config.bodyType || 'Normale'}
                                            onChange={(e) => setConfig({...config, bodyType: e.target.value})}
                                        >
                                            <option>Minuta</option>
                                            <option>Normale</option>
                                            <option>Sportiva</option>
                                            <option>Formosa</option>
                                            <option>In carne</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: '#94a3b8', fontSize: '10px' }}>▼</div>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Caratteristiche Fisiche */}
                            <div style={{ marginBottom: '16px' }}>
                                <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Caratteristiche Fisiche (occhiali, lentiggini, tatuaggi, ecc.)</label>
                                <input 
                                    style={{
                                      width: '100%',
                                      backgroundColor: 'rgba(255,255,255,0.75)',
                                      border: '1px solid rgba(226,232,240,0.6)',
                                      borderRadius: '12px',
                                      padding: '12px 16px',
                                      fontSize: '14px',
                                      color: '#1e293b',
                                      outline: 'none',
                                      fontWeight: 500,
                                      boxSizing: 'border-box',
                                      backdropFilter: 'blur(4px)'
                                    }}
                                    placeholder="Es: Occhiali eleganti, lentiggini, sorriso gentile..."
                                    value={config.physicalTraits}
                                    onChange={(e) => setConfig({...config, physicalTraits: e.target.value})}
                                />
                            </div>
                            
                            {/* Carattere & Personalità */}
                            <div>
                                <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Carattere & Personalità</label>
                                <textarea 
                                    style={{
                                      width: '100%',
                                      backgroundColor: 'rgba(255,255,255,0.75)',
                                      border: '1px solid rgba(226,232,240,0.6)',
                                      borderRadius: '16px',
                                      padding: '16px 20px',
                                      fontSize: '15px',
                                      color: '#1e293b',
                                      height: '80px',
                                      resize: 'none',
                                      outline: 'none',
                                      fontWeight: 500,
                                      lineHeight: 1.6,
                                      boxSizing: 'border-box',
                                      backdropFilter: 'blur(4px)'
                                    }}
                                    placeholder="Es: Empatica, calma, saggia, buona ascoltatrice..."
                                    value={config.personality}
                                    onChange={(e) => setConfig({...config, personality: e.target.value})}
                                />
                            </div>
                        </div>
                        
                        {/* Section 4: Modulazione Voce */}
                        <div style={{ 
                          marginBottom: '28px',
                          backgroundColor: 'rgba(147, 51, 234, 0.05)',
                          borderRadius: '16px',
                          padding: '20px',
                          border: '1px solid rgba(147, 51, 234, 0.1)'
                        }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#7c3aed', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                                <Music2 size={18} style={{ color: '#9333ea' }}/> Modulazione Voce
                            </h3>
                            
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                                {/* Energia: Calma -> Dinamica */}
                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                        <label style={{ color: '#64748b', fontSize: '11px', fontWeight: 600 }}>Energia</label>
                                        <span style={{ fontSize: '10px', fontWeight: 700, color: '#9333ea', backgroundColor: 'rgba(147, 51, 234, 0.1)', padding: '2px 8px', borderRadius: '8px' }}>
                                            {(config.voiceEnergy || 50) < 33 ? 'Calma' : (config.voiceEnergy || 50) < 66 ? 'Bilanciata' : 'Dinamica'}
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <span style={{ fontSize: '10px', color: '#94a3b8' }}>🧘</span>
                                        <input 
                                            type="range" 
                                            min="0" 
                                            max="100" 
                                            value={config.voiceEnergy || 50}
                                            onChange={(e) => setConfig({...config, voiceEnergy: parseInt(e.target.value)})}
                                            style={{ flex: 1, height: '6px', borderRadius: '3px', appearance: 'none', background: `linear-gradient(to right, #a78bfa ${config.voiceEnergy || 50}%, #e2e8f0 ${config.voiceEnergy || 50}%)`, cursor: 'pointer' }}
                                        />
                                        <span style={{ fontSize: '10px', color: '#94a3b8' }}>⚡</span>
                                    </div>
                                </div>
                                
                                {/* Tono: Caldo -> Professionale */}
                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                        <label style={{ color: '#64748b', fontSize: '11px', fontWeight: 600 }}>Tono</label>
                                        <span style={{ fontSize: '10px', fontWeight: 700, color: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', padding: '2px 8px', borderRadius: '8px' }}>
                                            {(config.voiceTone || 50) < 33 ? 'Caldo' : (config.voiceTone || 50) < 66 ? 'Neutro' : 'Professionale'}
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <span style={{ fontSize: '10px', color: '#94a3b8' }}>❤️</span>
                                        <input 
                                            type="range" 
                                            min="0" 
                                            max="100" 
                                            value={config.voiceTone || 50}
                                            onChange={(e) => setConfig({...config, voiceTone: parseInt(e.target.value)})}
                                            style={{ flex: 1, height: '6px', borderRadius: '3px', appearance: 'none', background: `linear-gradient(to right, #fbbf24 ${config.voiceTone || 50}%, #e2e8f0 ${config.voiceTone || 50}%)`, cursor: 'pointer' }}
                                        />
                                        <span style={{ fontSize: '10px', color: '#94a3b8' }}>💼</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Submit Button */}
                        <button 
                            onClick={handleConfigSubmit}
                            disabled={isGeneratingProfile}
                            style={{
                              width: '100%',
                              backgroundColor: isGeneratingProfile ? '#64748b' : '#0f172a',
                              color: 'white',
                              padding: '20px',
                              borderRadius: '16px',
                              fontWeight: 700,
                              fontSize: '16px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              gap: '12px',
                              border: 'none',
                              cursor: isGeneratingProfile ? 'not-allowed' : 'pointer',
                              boxShadow: '0 8px 24px rgba(15, 23, 42, 0.2)',
                              transition: 'all 0.2s'
                            }}
                        >
                            {isGeneratingProfile ? (
                                <>
                                    <Loader2 className="animate-spin" /> {loadingStep || 'Creazione in corso...'}
                                </>
                            ) : (
                                <>
                                    Crea il tuo Confidente <ArrowRight />
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
  }

  // --- MAIN CHAT INTERFACE (Light Theme) ---
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'row',
      height: '100vh',
      width: '100%',
      backgroundImage: "url('background.png')",
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      position: 'relative',
      fontFamily: 'Outfit, sans-serif',
      color: '#1e293b',
      overflow: 'hidden'
    }}>
      
      {/* LEFT COLUMN: PROFILE SIDEBAR - RESPONSIVE */}
      <aside style={{
        width: '320px',
        minWidth: '280px',
        maxWidth: '350px',
        backgroundColor: 'rgba(255,255,255,0.9)',
        backdropFilter: 'blur(12px)',
        borderRight: '1px solid rgba(226,232,240,0.6)',
        display: 'flex',
        flexDirection: 'column',
        padding: '20px',
        overflowY: 'auto',
        position: 'relative',
        zIndex: 10,
        flexShrink: 0
      }}>
        
        {/* Header: Logo + Progetto Confidente - CLICCABILE per tornare al menu */}
        <div 
          onClick={() => { if(window.confirm('Vuoi tornare al menu principale? La conversazione verrà terminata.')) { disconnect(); setIsConfigured(false); } }}
          style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', cursor: 'pointer', transition: 'opacity 0.2s' }}
          onMouseEnter={(e) => e.currentTarget.style.opacity = '0.7'}
          onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          title="Torna al menu principale"
        >
          <div style={{
            width: '40px',
            height: '40px',
            backgroundColor: 'rgba(255,255,255,0.95)',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 12px rgba(147, 112, 219, 0.2)',
            overflow: 'hidden',
            flexShrink: 0
          }}>
            <img src="logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
          </div>
          <div>
            <div style={{ fontSize: '8px', fontWeight: 700, letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>Progetto</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', letterSpacing: '-0.02em' }}>Confidente</div>
          </div>
        </div>

        {/* Assistant Name */}
        <h2 style={{
          fontSize: '22px',
          fontWeight: 700,
          color: '#0f172a',
          marginBottom: '2px',
          lineHeight: 1.2
        }}>
          {config.name || 'Il tuo Confidente'}
        </h2>
        <p style={{
          fontSize: '11px',
          fontWeight: 600,
          color: '#64748b',
          marginBottom: '12px'
        }}>
          Confidente di {config.userName || 'Te'}
        </p>

        {/* Avatar Photo - RESPONSIVE con aspect ratio */}
        <div style={{
          width: '100%',
          paddingBottom: '133%', /* 3:4 aspect ratio */
          position: 'relative',
          borderRadius: '12px',
          overflow: 'hidden',
          backgroundColor: '#f1f5f9',
          marginBottom: '12px',
          boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
          border: '2px solid white'
        }}>
          {avatarUrl ? (
            <img src={avatarUrl} alt="Avatar" style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%', 
              height: '100%', 
              objectFit: 'cover' 
            }} />
          ) : (
            <div style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%', 
              height: '100%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center' 
            }}>
              <User size={40} style={{ color: '#cbd5e1' }} />
            </div>
          )}
        </div>

        {/* Info Section - COMPATTA */}
        <div style={{
          backgroundColor: 'rgba(255,255,255,0.7)',
          borderRadius: '12px',
          padding: '14px',
          marginBottom: '12px',
          border: '1px solid rgba(226,232,240,0.5)'
        }}>
          {/* Age */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '8px', borderBottom: '1px solid #f1f5f9', marginBottom: '8px' }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Età</span>
            <span style={{ fontSize: '13px', fontWeight: 700, color: '#1e293b' }}>{config.age} anni</span>
          </div>
          
          {/* Biography */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '6px' }}>
              <Heart size={10} style={{ color: '#9333ea' }} />
              <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Biografia</span>
            </div>
            <p style={{
              fontSize: '12px',
              color: '#475569',
              lineHeight: 1.5,
              fontStyle: 'italic',
              backgroundColor: '#f8fafc',
              padding: '10px',
              borderRadius: '10px',
              maxHeight: '80px',
              overflowY: 'auto',
              margin: 0
            }}>
              "{config.biography || config.personality}"
            </p>
          </div>
        </div>

        {/* Spacer */}
        <div style={{ flex: 1, minHeight: '10px' }} />

        {/* Connect Button at bottom */}
        <div style={{ marginTop: 'auto' }}>
          {!isConnected ? (
            <button 
              onClick={connect}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px',
                padding: '14px 20px',
                backgroundColor: '#0f172a',
                color: 'white',
                borderRadius: '12px',
                fontWeight: 700,
                fontSize: '13px',
                border: 'none',
                cursor: 'pointer',
                boxShadow: '0 4px 16px rgba(15, 23, 42, 0.2)',
                transition: 'all 0.2s'
              }}
            >
              <Mic size={18} />
              INIZIA A PARLARE
            </button>
          ) : (
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                onClick={toggleMute}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '12px',
                  backgroundColor: isMuted ? '#fef2f2' : 'white',
                  color: isMuted ? '#ef4444' : '#475569',
                  borderRadius: '10px',
                  fontWeight: 600,
                  border: isMuted ? '1px solid #fecaca' : '1px solid #e2e8f0',
                  cursor: 'pointer'
                }}
              >
                {isMuted ? <MicOff size={18} /> : <Mic size={18} />}
              </button>
              <button 
                onClick={disconnect}
                style={{
                  flex: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  padding: '12px 16px',
                  backgroundColor: '#fef2f2',
                  color: '#ef4444',
                  borderRadius: '10px',
                  fontWeight: 700,
                  fontSize: '13px',
                  border: '1px solid #fecaca',
                  cursor: 'pointer'
                }}
              >
                <PhoneOff size={16} />
                Termina
              </button>
            </div>
          )}
          
          {/* Status indicator */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            gap: '6px', 
            marginTop: '12px',
            padding: '6px',
            backgroundColor: 'rgba(255,255,255,0.5)',
            borderRadius: '16px'
          }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: isConnected ? '#22c55e' : '#f87171',
              animation: isConnected ? 'pulse 2s infinite' : 'none'
            }} />
            <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {isConnected ? 'CONNESSO' : 'OFFLINE'}
            </span>
          </div>
          
          {/* Copyright */}
          <div style={{
            marginTop: '12px',
            fontSize: '8px',
            fontWeight: 500,
            color: '#94a3b8',
            textAlign: 'center',
            letterSpacing: '0.02em',
            lineHeight: 1.4
          }}>
            © Effetre Properties IA Division 2025<br/>All rights reserved
          </div>
        </div>
      </aside>

      {/* RIGHT COLUMN: CHAT AREA */}
      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        zIndex: 10,
        height: '100%',
        backgroundColor: 'rgba(255,255,255,0.3)'
      }}>
        
        {/* Error Message */}
        {error && (
          <div style={{
            position: 'absolute',
            top: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 50,
            padding: '12px 24px',
            backgroundColor: '#ef4444',
            color: 'white',
            borderRadius: '12px',
            boxShadow: '0 8px 24px rgba(239, 68, 68, 0.3)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: 500,
            fontSize: '14px'
          }}>
            <Info size={18} /> {error}
          </div>
        )}

        {/* Central Visualizer Area - MOLTO COMPATTO */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '16px',
          padding: '10px 20px',
          flexShrink: 0,
          borderBottom: '1px solid rgba(226,232,240,0.4)',
          backgroundColor: 'rgba(255,255,255,0.4)'
        }}>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', transform: 'scale(0.5)' }}>
            <AudioVisualizer isPlaying={isConnected} volume={audioVolume} />
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
              {isConnected ? (
                isMuted ? (
                  <MicOff size={28} style={{ color: '#cbd5e1' }} />
                ) : (
                  <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '50%', boxShadow: '0 2px 8px rgba(147, 112, 219, 0.2)' }}>
                    <Mic size={28} style={{ color: '#9333ea' }} />
                  </div>
                )
              ) : (
                <PhoneOff size={28} style={{ color: '#e2e8f0' }} />
              )}
            </div>
          </div>
          <p style={{
            margin: 0,
            color: '#94a3b8',
            fontSize: '11px',
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase'
          }}>
            {isConnected 
              ? isMuted ? "Mic OFF" : `In conversazione con ${config.name}` 
              : "Premi 'Inizia a parlare'"}
          </p>
        </div>

        {/* Transcript Area */}
        <div 
          ref={transcriptRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
            minHeight: 0
          }}
        >
          {transcripts.length === 0 && (
            <div style={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#cbd5e1',
              gap: '8px',
              minHeight: '150px'
            }}>
              <Sparkles size={32} />
              <span style={{ fontSize: '14px', fontWeight: 500 }}>Inizia la conversazione...</span>
            </div>
          )}
          
          {transcripts.map((t) => (
            <div 
              key={t.id} 
              style={{ 
                display: 'flex', 
                justifyContent: t.sender === 'user' ? 'flex-end' : 'flex-start' 
              }}
            >
              <div style={{
                maxWidth: '70%',
                borderRadius: '16px',
                padding: t.type === 'action' ? '0' : '16px 20px',
                fontSize: '14px',
                lineHeight: 1.6,
                backgroundColor: t.sender === 'user' 
                  ? '#0f172a' 
                  : t.type === 'action' 
                    ? 'transparent' 
                    : 'white',
                color: t.sender === 'user' ? 'white' : '#334155',
                boxShadow: t.type === 'action' ? 'none' : '0 2px 12px rgba(0,0,0,0.08)',
                border: t.sender === 'user' || t.type === 'action' ? 'none' : '1px solid #f1f5f9'
              }}>
                {/* Label */}
                {t.type !== 'action' && (
                  <div style={{
                    fontSize: '10px',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    marginBottom: '6px',
                    color: t.sender === 'user' ? '#94a3b8' : '#9333ea'
                  }}>
                    {t.sender === 'user' ? (config.userName || 'Tu') : config.name}
                  </div>
                )}

                {/* Text */}
                {t.type === 'text' && <div>{t.text}</div>}

                {/* Image - Ridimensionata */}
                {t.type === 'image' && t.image && (
                  <div style={{ 
                    marginTop: '8px', 
                    borderRadius: '12px', 
                    overflow: 'hidden', 
                    position: 'relative',
                    maxWidth: '300px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                  }}>
                    <img 
                      src={t.image} 
                      alt="Foto" 
                      style={{ 
                        width: '100%', 
                        height: 'auto',
                        maxHeight: '400px',
                        objectFit: 'cover',
                        display: 'block'
                      }} 
                    />
                    <button 
                      onClick={() => t.image && downloadImage(t.image, `foto-${t.sender === 'user' ? config.userName : config.name}-${Date.now()}.png`)}
                      style={{
                        position: 'absolute',
                        bottom: '8px',
                        right: '8px',
                        padding: '8px',
                        backgroundColor: 'white',
                        color: '#1e293b',
                        borderRadius: '50%',
                        border: 'none',
                        cursor: 'pointer',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                      title="Scarica immagine"
                    >
                      <Download size={14} />
                    </button>
                  </div>
                )}

                {/* Action */}
                {t.type === 'action' && t.actionUrl && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ fontSize: '12px', fontWeight: 500, color: '#64748b' }}>{t.text}</div>
                    <a 
                      href={t.actionUrl} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        padding: '16px',
                        borderRadius: '12px',
                        fontWeight: 700,
                        color: 'white',
                        textDecoration: 'none',
                        background: t.actionIcon === 'mail' 
                          ? 'linear-gradient(135deg, #ec4899, #f43f5e)' 
                          : 'linear-gradient(135deg, #10b981, #14b8a6)',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.15)'
                      }}
                    >
                      <div style={{ padding: '8px', backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: '50%' }}>
                        {t.actionIcon === 'mail' ? <Mail size={20} /> : <MessageCircle size={20} />}
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '15px' }}>{t.actionLabel}</span>
                        <span style={{ fontSize: '10px', opacity: 0.8, fontWeight: 400 }}>Clicca per aprire</span>
                      </div>
                      <ExternalLink size={16} style={{ marginLeft: 'auto', opacity: 0.8 }} />
                    </a>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Input Area per foto + Footer */}
        <div style={{
          borderTop: '1px solid rgba(226,232,240,0.5)',
          backgroundColor: 'rgba(255,255,255,0.7)',
          backdropFilter: 'blur(8px)'
        }}>
          {/* Photo Upload Area */}
          <div style={{
            padding: '12px 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px'
          }}>
            <input 
              type="file" 
              ref={fileInputRef}
              accept="image/*"
              onChange={handleUserPhotoUpload}
              style={{ display: 'none' }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzingPhoto}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 20px',
                backgroundColor: isAnalyzingPhoto ? '#94a3b8' : 'white',
                color: isAnalyzingPhoto ? 'white' : '#64748b',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: isAnalyzingPhoto ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
              }}
            >
              {isAnalyzingPhoto ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> Analizzo la foto...
                </>
              ) : (
                <>
                  <Camera size={16} /> Invia una foto
                </>
              )}
            </button>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>
              Condividi un'immagine con {config.name}
            </span>
          </div>
          
          {/* Footer */}
          <div style={{
            padding: '12px 16px',
            textAlign: 'center',
            fontSize: '10px',
            fontWeight: 700,
            color: '#cbd5e1',
            letterSpacing: '0.2em',
            textTransform: 'uppercase'
          }}>
            Progetto Confidente • AI Division
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;