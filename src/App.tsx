import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type, Tool } from '@google/genai';
import { TranscriptItem, AssistantConfig } from './types';
import { createBlob, decode, decodeAudioData } from './utils/audio';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Mic, MicOff, PhoneOff, User, Bot, Sparkles, Image as ImageIcon, ArrowRight, Loader2, Heart, Info, Mail, MessageCircle, ExternalLink, Download, Wand2, UserCircle, Sliders, Music2, Menu } from 'lucide-react';

const LIVE_MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const IMAGE_MODEL_NAME = 'gemini-2.5-flash-image';
const TEXT_MODEL_NAME = 'gemini-3-flash-preview';

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

// --- BRANDING COMPONENT (Updated to match photo style) ---
const AppLogo = ({ size = 48, className = "" }: { size?: number, className?: string }) => {
  const [imgError, setImgError] = useState(false);

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <div className="relative h-full w-full bg-white border border-slate-100 rounded-[1.2rem] flex items-center justify-center overflow-hidden shadow-lg shadow-purple-100">
        {!imgError ? (
           <img 
             src="logo.png" 
             alt="Logo" 
             className="w-full h-full object-cover"
             onError={() => setImgError(true)}
           />
        ) : (
          <div className="relative z-10 flex items-center justify-center p-2">
            <Heart size={size * 0.6} className="text-purple-400 fill-purple-400 absolute left-2 opacity-90" />
            <Heart size={size * 0.6} className="text-amber-400 fill-amber-400 absolute right-2 top-2 opacity-90 mix-blend-multiply" />
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
    physicalTraits: 'Sorriso gentile, occhiali eleganti',
    personality: 'Empatica, calma, saggia, buona ascoltatrice',
    name: '',
    biography: '',
    visualPrompt: '',
    voicePitch: 0,
    voiceSpeed: 1.0,
  });
  const [isConfigured, setIsConfigured] = useState(false);
  const [isGeneratingProfile, setIsGeneratingProfile] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);

  // App State
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [audioVolume, setAudioVolume] = useState(0);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    if (process.env.API_KEY) {
      aiRef.current = new GoogleGenAI({ apiKey: process.env.API_KEY });
    } else {
      setError("API Key non trovata.");
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

  const handleConfigSubmit = async () => {
    if (!aiRef.current) return;
    setIsGeneratingProfile(true);
    setError(null);

    try {
        const hasManualName = config.name && config.name.trim().length > 0;
        setLoadingStep(hasManualName ? `Sto definendo la personalità di ${config.name}...` : 'Sto creando il tuo amico ideale...');
        
        const basePrompt = `Crea un profilo per un COMPAGNO UMANO: Genere ${config.gender}, Età ${config.age}, Capelli ${config.hairColor}, Occhi ${config.eyeColor}, Pelle ${config.skinTone}, Tratti ${config.physicalTraits}, Personalità ${config.personality}.`;
        const nameInstruction = hasManualName ? `Il nome è "${config.name}".` : `Inventa un nome.`;

        const profilePrompt = `${basePrompt} ${nameInstruction} Rispondi JSON: {name, biography, visualPrompt}`;
        
        const textResponse = await aiRef.current.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: profilePrompt,
            config: { responseMimeType: 'application/json', responseSchema: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, biography: { type: Type.STRING }, visualPrompt: { type: Type.STRING } }, required: ['name', 'biography', 'visualPrompt'] } }
        });

        const profileData = JSON.parse(textResponse.text || '{}');
        if (!profileData.name) throw new Error("Errore generazione profilo.");

        setConfig(prev => ({ ...prev, name: profileData.name, biography: profileData.biography, visualPrompt: profileData.visualPrompt }));
        setLoadingStep(`Sto scattando una foto a ${profileData.name}...`);
        
        const imageResponse = await aiRef.current.models.generateContent({
            model: IMAGE_MODEL_NAME,
            contents: { parts: [{ text: "Close-up portrait, 8k, photorealistic, cinematic lighting, " + profileData.visualPrompt }] },
        });

        let foundUrl: string | null = null;
        if (imageResponse.candidates?.[0]?.content?.parts) {
            for (const part of imageResponse.candidates[0].content.parts) {
                if (part.inlineData) {
                    foundUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                    break;
                }
            }
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
    try {
        addTranscript({ sender: 'model', type: 'text', text: isSelfie ? `📸 Un attimo, mi metto in posa...` : `🎨 Sto disegnando: "${prompt}"...`, isComplete: true });
        
        let finalPrompt = prompt;
        if (isSelfie && config.visualPrompt) {
            finalPrompt = `PHOTOREALISTIC RAW PHOTO, 8k. SUBJECT: ${config.visualPrompt}. ACTION: ${prompt}. STYLE: Authentic.`;
        }

        const response = await aiRef.current.models.generateContent({
            model: IMAGE_MODEL_NAME,
            contents: { parts: [{ text: finalPrompt }] },
        });

        let imageUrl: string | null = null;
        if (response.candidates?.[0]?.content?.parts) {
            for (const part of response.candidates[0].content.parts) {
                if (part.inlineData) {
                    imageUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                    break;
                }
            }
        }
        
        if (imageUrl) {
            addTranscript({ sender: 'model', type: 'image', image: imageUrl, isComplete: true });
            return "Foto scattata.";
        }
        return "Errore fotocamera.";
    } catch (e) {
        return "Errore tecnico.";
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
    if (!aiRef.current) return;
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
          systemInstruction: `Sei ${config.name}, amico di ${config.userName}. Personalità: ${config.personality}. Bio: ${config.biography}. Sii naturale, empatico. Non dire di essere un'AI.`,
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
        <div 
          className="min-h-screen bg-[#FDFCF8] relative overflow-hidden font-sans text-slate-800 selection:bg-purple-200 bg-cover bg-center bg-no-repeat transition-all duration-700"
          style={{ backgroundImage: "url('background.png')" }}
        >
            {/* Overlay for better readability if background is busy */}
            <div className="absolute inset-0 bg-white/40 pointer-events-none" />

            <div className="container mx-auto max-w-6xl p-6 md:p-8 relative z-10 flex flex-col lg:flex-row gap-12 h-full lg:h-screen lg:items-center">
                
                {/* Left Side: Brand & Description */}
                <div className="w-full lg:w-5/12 flex flex-col justify-center">
                    <div className="flex items-center gap-4 mb-8">
                        <AppLogo size={68} className="shadow-xl shadow-purple-100/50 rounded-[1.2rem]" />
                        <div className="flex flex-col">
                            <span className="text-[10px] font-bold tracking-[0.25em] text-slate-500 uppercase mb-1">Progetto</span>
                            <span className="text-xl font-bold text-slate-800 tracking-tight leading-none">PARLIAMO</span>
                        </div>
                    </div>

                    <h1 className="text-5xl md:text-6xl font-bold text-slate-900 mb-6 leading-[1.1] tracking-tight">
                        Parliamo...
                    </h1>
                    
                    <p className="text-lg text-slate-700 font-medium leading-relaxed max-w-md mb-8">
                        Sono qualcuno che ti ascolta davvero. 
                        Configurami, dammi un volto e una voce, e parliamo di tutto ciò che ti passa per la testa.
                    </p>

                    {/* Feature Pill */}
                    <div className="inline-flex items-center gap-4 p-4 pr-8 rounded-2xl bg-white/60 border border-white/80 shadow-sm backdrop-blur-md w-fit">
                        <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center text-purple-600">
                            <Heart fill="currentColor" size={20} />
                        </div>
                        <span className="font-semibold text-slate-700">Ascolto Attivo</span>
                    </div>
                </div>

                {/* Right Side: Configuration Form */}
                <div className="w-full lg:w-7/12">
                    <div className="bg-white/40 backdrop-blur-sm rounded-[2rem] border border-white/60 shadow-xl shadow-purple-100/30 p-1">
                        <div className="bg-white/50 rounded-[1.8rem] p-6 md:p-10 max-h-[85vh] overflow-y-auto custom-scrollbar">
                            
                            {error && (
                                <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm flex items-center gap-2">
                                    <Info size={16} /> {error}
                                </div>
                            )}

                            <div className="space-y-10">
                                {/* Section 1 */}
                                <div>
                                    <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
                                        <User size={20} className="text-slate-400"/> Chi sei tu?
                                    </h3>
                                    <div>
                                        <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Il tuo Nome</label>
                                        <input 
                                            className="w-full bg-slate-100/80 border border-slate-200 focus:border-purple-300 focus:bg-white focus:ring-4 focus:ring-purple-100/50 rounded-2xl px-5 py-4 text-slate-800 placeholder-slate-400 outline-none transition-all font-medium"
                                            placeholder="Come vuoi che ti chiami?"
                                            value={config.userName}
                                            onChange={(e) => setConfig({...config, userName: e.target.value})}
                                        />
                                    </div>
                                </div>

                                {/* Section 2 */}
                                <div>
                                    <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
                                        <Bot size={20} className="text-amber-500"/> Il tuo Confidente
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
                                        <div>
                                            <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Genere Assistente</label>
                                            <div className="relative">
                                                <select 
                                                    className="w-full bg-slate-100/80 border border-slate-200 focus:border-purple-300 focus:bg-white focus:ring-4 focus:ring-purple-100/50 rounded-2xl px-5 py-4 text-slate-800 appearance-none outline-none cursor-pointer font-medium"
                                                    value={config.gender}
                                                    onChange={(e) => setConfig({...config, gender: e.target.value})}
                                                >
                                                    <option>Uomo</option>
                                                    <option>Donna</option>
                                                    <option>Non-binary</option>
                                                </select>
                                                <div className="absolute right-5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">▼</div>
                                            </div>
                                        </div>
                                        <div>
                                            <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Età Apparente</label>
                                            <input 
                                                type="number"
                                                className="w-full bg-slate-100/80 border border-slate-200 focus:border-purple-300 focus:bg-white focus:ring-4 focus:ring-purple-100/50 rounded-2xl px-5 py-4 text-slate-800 outline-none font-medium"
                                                value={config.age}
                                                onChange={(e) => setConfig({...config, age: e.target.value})}
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Nome Assistente (Opzionale)</label>
                                        <input 
                                            className="w-full bg-slate-100/80 border border-slate-200 focus:border-purple-300 focus:bg-white focus:ring-4 focus:ring-purple-100/50 rounded-2xl px-5 py-4 text-slate-800 placeholder-slate-400 outline-none transition-all font-medium"
                                            placeholder="Lascia vuoto per generare automaticamente"
                                            value={config.name}
                                            onChange={(e) => setConfig({...config, name: e.target.value})}
                                        />
                                    </div>
                                </div>

                                {/* Voice Modulation */}
                                <div className="bg-purple-50/50 rounded-2xl p-5 border border-purple-100/50">
                                    <h4 className="text-purple-800 text-xs uppercase tracking-wider font-bold mb-4 flex items-center gap-2">
                                        <Sliders size={14} className="text-purple-500" /> Modulazione Vocale
                                    </h4>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                        <div>
                                            <label className="flex justify-between text-slate-500 text-xs font-semibold mb-3">
                                                <span>Tonalità</span>
                                                <span className="text-purple-600 bg-purple-100 px-2 py-0.5 rounded text-[10px]">{config.voicePitch || 0}</span>
                                            </label>
                                            <input type="range" min="-200" max="200" step="10" value={config.voicePitch || 0} onChange={(e) => setConfig({...config, voicePitch: parseInt(e.target.value)})} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-purple-500" />
                                        </div>
                                        <div>
                                            <label className="flex justify-between text-slate-500 text-xs font-semibold mb-3">
                                                <span>Velocità</span>
                                                <span className="text-purple-600 bg-purple-100 px-2 py-0.5 rounded text-[10px]">x{config.voiceSpeed || 1.0}</span>
                                            </label>
                                            <input type="range" min="0.85" max="1.15" step="0.05" value={config.voiceSpeed || 1.0} onChange={(e) => setConfig({...config, voiceSpeed: parseFloat(e.target.value)})} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500" />
                                        </div>
                                    </div>
                                </div>

                                {/* Section 3 */}
                                <div>
                                    <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
                                        <Wand2 size={20} className="text-slate-400"/> Dettagli
                                    </h3>
                                    <div className="grid grid-cols-3 gap-3 mb-5">
                                        <div>
                                            <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Capelli</label>
                                            <input className="w-full bg-slate-100/80 border border-slate-200 focus:bg-white focus:ring-2 focus:ring-purple-100 rounded-xl px-4 py-3 text-slate-800 text-sm outline-none font-medium" placeholder="Castani" value={config.hairColor} onChange={(e) => setConfig({...config, hairColor: e.target.value})} />
                                        </div>
                                        <div>
                                            <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Occhi</label>
                                            <input className="w-full bg-slate-100/80 border border-slate-200 focus:bg-white focus:ring-2 focus:ring-purple-100 rounded-xl px-4 py-3 text-slate-800 text-sm outline-none font-medium" placeholder="Verdi" value={config.eyeColor} onChange={(e) => setConfig({...config, eyeColor: e.target.value})} />
                                        </div>
                                        <div>
                                            <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Pelle</label>
                                            <input className="w-full bg-slate-100/80 border border-slate-200 focus:bg-white focus:ring-2 focus:ring-purple-100 rounded-xl px-4 py-3 text-slate-800 text-sm outline-none font-medium" placeholder="Chiara" value={config.skinTone} onChange={(e) => setConfig({...config, skinTone: e.target.value})} />
                                        </div>
                                    </div>
                                    <div>
                                        <label className="block text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-2 ml-1">Carattere & Personalità</label>
                                        <textarea 
                                            className="w-full bg-slate-100/80 border border-slate-200 focus:border-purple-300 focus:bg-white focus:ring-4 focus:ring-purple-100/50 rounded-2xl p-5 text-slate-800 h-24 resize-none outline-none font-medium leading-relaxed"
                                            placeholder="Descrivi come si comporta..."
                                            value={config.personality}
                                            onChange={(e) => setConfig({...config, personality: e.target.value})}
                                        />
                                    </div>
                                </div>

                                <button 
                                    onClick={handleConfigSubmit}
                                    disabled={isGeneratingProfile}
                                    className="w-full bg-slate-900 hover:bg-slate-800 text-white py-5 rounded-2xl font-bold text-lg flex items-center justify-center gap-3 transition-all transform hover:scale-[1.01] active:scale-[0.99] shadow-xl shadow-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isGeneratingProfile ? (
                                        <>
                                            <Loader2 className="animate-spin" /> {loadingStep || 'Creazione in corso...'}
                                        </>
                                    ) : (
                                        <>
                                            Crea il tuo Amico <ArrowRight />
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
  }

  // --- MAIN CHAT INTERFACE (Light Theme) ---
  return (
    <div 
        className="flex flex-col md:flex-row h-screen w-full bg-[#FDFCF8] overflow-hidden relative font-sans text-slate-800 bg-cover bg-center"
        style={{ backgroundImage: "url('background.png')" }}
    >
      
      {/* Overlay for readability */}
      <div className="absolute inset-0 bg-white/60 pointer-events-none z-0" />

      {/* LEFT COLUMN: PROFILE */}
      <aside className="w-full md:w-80 lg:w-96 bg-white/80 backdrop-blur-xl border-r border-slate-200/60 flex flex-col p-6 z-10 overflow-y-auto hidden md:flex shadow-sm">
         <div className="mb-8 flex items-center gap-3">
            <AppLogo size={40} className="shadow-md shadow-purple-100" />
            <h1 className="text-lg font-bold text-slate-800 tracking-tight">Parliamo...</h1>
         </div>

         {/* Profile Card */}
         <div className="flex flex-col gap-4 bg-white p-5 rounded-[1.5rem] border border-slate-100 shadow-xl shadow-slate-200/50">
             <div className="text-center mb-1">
                 <h2 className="text-2xl font-bold text-slate-800">
                     {config.name || (config.gender === 'Donna' ? 'La tua Amica' : 'Il tuo Amico')}
                 </h2>
                 <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mt-1">Confidente di {config.userName || 'Te'}</p>
             </div>
             
             <div className="w-full aspect-square rounded-2xl overflow-hidden border-4 border-slate-50 bg-slate-100 relative shadow-inner">
                 {avatarUrl ? (
                     <img src={avatarUrl} alt="Avatar" className="w-full h-full object-cover" />
                 ) : (
                     <div className="w-full h-full flex items-center justify-center">
                         <User size={48} className="text-slate-300" />
                     </div>
                 )}
             </div>

             <div className="space-y-4 mt-2">
                 <div className="flex justify-between items-center border-b border-slate-50 pb-3">
                     <span className="text-slate-400 text-xs font-bold uppercase">Età</span>
                     <span className="text-slate-700 font-bold">{config.age} anni</span>
                 </div>
                 
                 <div className="flex flex-col gap-2">
                     <span className="text-slate-400 text-xs font-bold uppercase flex items-center gap-1">
                        <Heart size={12} className="text-purple-500" /> Biografia
                     </span>
                     <p className="text-slate-600 text-sm leading-relaxed italic bg-slate-50 p-4 rounded-xl max-h-40 overflow-y-auto border border-slate-100">
                         "{config.biography || config.personality}"
                     </p>
                 </div>
             </div>
         </div>
         
         <div className="mt-auto pt-6 grid grid-cols-2 gap-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
             <div className="flex items-center gap-1 hover:text-purple-500 cursor-pointer transition-colors"><Mail size={12}/> Email Support</div>
             <div className="flex items-center gap-1 hover:text-green-500 cursor-pointer transition-colors"><MessageCircle size={12}/> WhatsApp Support</div>
         </div>
      </aside>

      {/* RIGHT COLUMN: CHAT & ACTIONS */}
      <main className="flex-1 flex flex-col relative z-10 h-full">
          {/* Header Status */}
          <div className="w-full p-4 flex justify-between md:justify-end items-center border-b border-slate-100 bg-white/50 backdrop-blur-sm shrink-0">
            <div className="md:hidden flex items-center gap-2">
                 <AppLogo size={32} />
                 <span className="font-bold text-slate-800">{config.name}</span>
            </div>
            <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{isConnected ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="absolute top-20 left-1/2 transform -translate-x-1/2 z-50 px-6 py-3 bg-red-500 text-white rounded-xl shadow-xl flex items-center gap-2 font-medium text-sm">
                <Info size={18} /> {error}
            </div>
          )}

          {/* Central Area: Visualizer (Fixed Height now to allow chat to scroll) */}
          <div className="shrink-0 flex flex-col items-center justify-center py-6 relative">
               <div className="relative flex items-center justify-center scale-90 md:scale-100">
                   <AudioVisualizer isPlaying={isConnected} volume={audioVolume} />
                   <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        {isConnected ? (
                             isMuted ? <MicOff size={40} className="text-slate-300" /> : <div className="p-4 bg-white rounded-full shadow-lg shadow-purple-100"><Mic size={40} className="text-purple-500 animate-pulse" /></div>
                        ) : (
                            <div className="text-slate-200">
                                <PhoneOff size={40} />
                            </div>
                        )}
                   </div>
               </div>
               <p className="mt-4 text-slate-400 text-sm font-semibold tracking-widest uppercase animate-fade-in">
                {isConnected 
                    ? isMuted ? "Microfono disattivato" : `Parla con ${config.name}...` 
                    : "In pausa"}
               </p>
          </div>

          {/* Transcript Scroll Area - Updated for Full Height Scrolling */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gradient-to-t from-white via-white/80 to-transparent min-h-0" ref={transcriptRef}>
              {transcripts.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-slate-300 space-y-2 min-h-[100px]">
                    <Sparkles size={24} />
                    <span className="text-sm font-medium">Inizia la conversazione...</span>
                </div>
              )}
              {transcripts.map((t) => (
                <div key={t.id} className={`flex ${t.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[90%] md:max-w-[70%] rounded-2xl px-5 py-4 text-sm leading-relaxed shadow-sm ${
                        t.sender === 'user' 
                        ? 'bg-slate-800 text-white rounded-tr-sm' 
                        : t.type === 'action' 
                            ? 'bg-transparent shadow-none p-0' 
                            : 'bg-white border border-slate-100 text-slate-700 rounded-tl-sm shadow-md shadow-slate-100'
                    }`}>
                        {/* Label */}
                        {t.type !== 'action' && (
                            <div className={`text-[10px] font-bold uppercase tracking-widest mb-1.5 ${t.sender === 'user' ? 'text-slate-400' : 'text-purple-500'}`}>
                                {t.sender === 'user' ? (config.userName || 'Tu') : config.name}
                            </div>
                        )}

                        {/* Text Content */}
                        {t.type === 'text' && <div>{t.text}</div>}

                        {/* Image Content */}
                        {t.type === 'image' && t.image && (
                            <div className="mt-2 rounded-xl overflow-hidden shadow-lg border-4 border-white relative group">
                                <img src={t.image} alt="Generata dall'AI" className="w-full h-auto object-cover" />
                                <button 
                                  onClick={() => t.image && downloadImage(t.image, `foto-${config.name}.png`)}
                                  className="absolute bottom-3 right-3 p-2 bg-white text-slate-800 rounded-full opacity-0 group-hover:opacity-100 transition-all shadow-md hover:scale-110"
                                  title="Scarica"
                                >
                                  <Download size={16} />
                                </button>
                            </div>
                        )}

                        {/* Action Buttons */}
                        {t.type === 'action' && t.actionUrl && (
                            <div className="flex flex-col gap-2 mt-1">
                                <div className="text-xs font-medium text-slate-500 ml-1">{t.text}</div>
                                <a 
                                    href={t.actionUrl} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className={`flex items-center gap-3 p-4 rounded-xl transition-all font-bold text-white shadow-lg transform hover:-translate-y-1 hover:shadow-xl ${
                                        t.actionIcon === 'mail' 
                                            ? 'bg-gradient-to-r from-pink-500 to-rose-500' 
                                            : 'bg-gradient-to-r from-emerald-500 to-teal-500'
                                    }`}
                                >
                                    <div className="p-2 bg-white/20 rounded-full">
                                        {t.actionIcon === 'mail' ? <Mail size={20} /> : <MessageCircle size={20} />}
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-base">{t.actionLabel}</span>
                                        <span className="text-[10px] opacity-80 font-normal">Clicca per aprire l'app</span>
                                    </div>
                                    <ExternalLink size={16} className="ml-auto opacity-80" />
                                </a>
                            </div>
                        )}
                    </div>
                </div>
             ))}
          </div>

          {/* Controls Footer */}
          <div className="bg-white border-t border-slate-100 z-20 flex flex-col items-center w-full shadow-[0_-10px_40px_-15px_rgba(0,0,0,0.05)] shrink-0">
            <div className="p-6 flex justify-center items-center gap-6 w-full">
                {!isConnected ? (
                    <button 
                        onClick={connect}
                        className="flex items-center gap-3 px-10 py-4 bg-slate-900 hover:bg-slate-800 text-white rounded-full font-bold tracking-wide shadow-xl shadow-slate-300 transition-all transform hover:scale-105 active:scale-95"
                    >
                        <Mic className="w-5 h-5" />
                        INIZIA A PARLARE
                    </button>
                ) : (
                    <>
                        <button 
                            onClick={toggleMute}
                            className={`p-5 rounded-full transition-all border shadow-lg ${isMuted ? 'bg-red-50 border-red-200 text-red-500' : 'bg-white border-slate-100 text-slate-700 hover:bg-slate-50'}`}
                        >
                            {isMuted ? <MicOff /> : <Mic />}
                        </button>
                        
                        <button 
                            onClick={disconnect}
                            className="px-8 py-4 bg-red-50 border border-red-100 text-red-500 hover:bg-red-500 hover:text-white rounded-full font-bold transition-all shadow-md"
                        >
                            Termina
                        </button>
                    </>
                )}
            </div>
            <div className="pb-4 text-[10px] font-bold text-slate-300 tracking-[0.2em] uppercase">
                Effetre Properties AI Division
            </div>
          </div>
      </main>
    </div>
  );
};

export default App;