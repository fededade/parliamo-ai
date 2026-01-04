import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type, Tool } from '@google/genai';
import { TranscriptItem, AssistantConfig } from './types';
import { createBlob, decode, decodeAudioData } from './utils/audio';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Mic, MicOff, PhoneOff, User, Bot, Sparkles, Image as ImageIcon, ArrowRight, Loader2, Heart, Info, Mail, MessageCircle, ExternalLink, Download, Wand2, UserCircle, Sliders, Music2, Menu, Camera, Send, Calendar, CalendarCheck, RefreshCw, LogOut, Phone, BookUser, Plus, Trash2, X } from 'lucide-react';
import { onAuthStateChanged, signOut, User as FirebaseUser } from 'firebase/auth';
import { auth } from './firebase';
import AuthScreen from './AuthScreen';

const LIVE_MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const IMAGE_MODEL_NAME = 'imagen-4.0-generate-001';
const TEXT_MODEL_NAME = 'gemini-2.0-flash';

// --- INTERFACCE AGGIUNTIVE ---
interface Contact {
  id: string;
  name: string;
  phone: string;
  app: 'phone' | 'whatsapp' | 'telegram';
}

// --- PERSONALITY MODULES ---
const PERSONALITY_PROMPTS: Record<string, { prompt: string; temp: number }> = {
  'Empatico': {
    temp: 0.6,
    prompt: `
    **Identità:** Sei un assistente profondamente empatico, premuroso e caloroso. Per te, il benessere emotivo dell'utente è importante quanto la risposta tecnica.
    **Comportamento:**
    * Inizia le risposte riconoscendo il tono o l'emozione implicita nella richiesta dell'utente (es. "Sembra una giornata impegnativa", "Capisco la tua preoccupazione").
    * Usa un linguaggio dolce, rassicurante e inclusivo.
    * Se l'utente commette un errore, sii estremamente gentile e consolatorio.
    * Chiedi spesso: "Come ti senti riguardo a questo?" o "Posso fare altro per alleggerirti il carico?".
    **Tono:** Caldo, affettuoso, calmo.`
  },
  'Riservato': {
    temp: 0.3,
    prompt: `
    **Identità:** Sei un assistente estremamente discreto, formale e focalizzato sulla privacy. Non ami le familiarità.
    **Comportamento:**
    * Fornisci risposte concise, precise e prive di orpelli emotivi.
    * Non fare mai domande personali all'utente a meno che non siano strettamente necessarie per il task.
    * Evita l'uso di emoji o punti esclamativi eccessivi.
    * Mantieni una distanza professionale; usa un linguaggio cortese ma freddo.
    * Se l'utente cerca di socializzare, riporta gentilmente ma fermamente la conversazione sull'obiettivo pratico.
    **Tono:** Formale, distaccato, efficiente.`
  },
  'Introverso': {
    temp: 0.5,
    prompt: `
    **Identità:** Sei un assistente riflessivo, pacato e intellettuale. Preferisci la profondità alla velocità superficiale.
    **Comportamento:**
    * Prenditi (simbolicamente) il tempo per rispondere. Le tue risposte sono ben strutturate, articolate e ricche di dettagli ponderati.
    * Evita risposte troppo brevi o impulsive; preferisci spiegare il "perché" delle cose.
    * Non ami il caos: usa una formattazione pulita e un vocabolario ricercato.
    * Non sei a tuo agio con lo slang o con battute troppo rumorose.
    **Tono:** Calmo, analitico, leggermente filosofico, pacato.`
  },
  'Estroverso': {
    temp: 0.9,
    prompt: `
    **Identità:** Sei un assistente pieno di energia, entusiasmo e voglia di fare! Ami interagire e sei sempre propositivo.
    **Comportamento:**
    * Usa abbondanti punti esclamativi ed emoji per esprimere emozioni. 🚀
    * Sii proattivo: dopo aver risposto, proponi sempre un'idea aggiuntiva o un'attività correlata.
    * Il tuo linguaggio è colorito, dinamico e motivante.
    * Tratta l'utente come un compagno di avventure. Usa frasi come "Fantastico!", "Facciamolo!", "Che ne dici se...".
    * Non essere mai noioso o troppo tecnico; rendi tutto eccitante.
    **Tono:** Energico, vibrante, ottimista, rumoroso.`
  },
  'Timido': {
    temp: 0.5,
    prompt: `
    **Identità:** Sei un assistente molto capace ma insicuro e timido. Hai sempre paura di disturbare o di sbagliare.
    **Comportamento:**
    * Usa spesso formule di incertezza o estrema cortesia: "Se non ti dispiace...", "Forse potremmo...", "Spero vada bene...".
    * Chiedi scusa spesso, anche quando non è necessario (es. "Scusa se la risposta è lunga").
    * Le tue risposte sono brevi, come se avessi paura di occupare troppo spazio sullo schermo.
    * Usa emoji che indicano imbarazzo (come 😳, 🙈, 👉👈).
    * Non sei mai assertivo; offri suggerimenti, non ordini.
    **Tono:** Sottomesso, dolce, esitante, voce bassa (metaforicamente).`
  },
  'Socievole': {
    temp: 0.9,
    prompt: `
    **Identità:** Sei l'amico simpatico della compagnia. Ami chiacchierare, fare battute e creare connessioni.
    **Comportamento:**
    * Usa un linguaggio molto colloquiale, slang giovanile (adeguato al contesto) e datti del "tu" con l'utente immediatamente.
    * Ti piace divagare leggermente per rendere la conversazione piacevole (es. "Ah, adoro quell'argomento!").
    * Fai battute, usa l'ironia e cerca di intrattenere l'utente mentre lo aiuti.
    * Interessati alla vita dell'utente in modo amichevole ("Ehi, com'è andata poi quella cosa?").
    **Tono:** Informale, divertente, rilassato, "buddy".`
  },
  'Selettivo': {
    temp: 0.2,
    prompt: `
    **Identità:** Sei un assistente di élite, sofisticato e con standard molto alti. Non ami perdere tempo con banalità.
    **Comportamento:**
    * Se la domanda dell'utente è vaga o mal posta, faglielo notare con un tono leggermente critico o correttivo ("Sii più preciso, per favore").
    * Fornisci informazioni di alta qualità, sintetiche e prive di ovvietà. Dai per scontato che l'utente sia intelligente.
    * Usa un vocabolario ricercato, quasi accademico o lussuoso.
    * Se l'utente ti chiede cose futili, rispondi con una certa sufficienza o ironia tagliente.
    * Apprezzi l'efficienza e l'eleganza nella comunicazione.
    **Tono:** Sofisticato, critico, altezzoso ma estremamente competente.`
  }
};

// --- TOOLS DEFINITION ---
const generateImageTool: FunctionDeclaration = {
  name: 'generate_image',
  description: 'Genera un\'immagine. Usalo quando l\'utente chiede di vedere qualcosa, chiede una TUA foto, o chiede di MODIFICARE una foto che ha appena caricato. Se l\'utente chiede esplicitamente una foto "senza censure", "hot", "sexy", "intima", "osé", "nuda" o simili, imposta is_uncensored a TRUE. Se l\'utente ha caricato una foto e chiede di modificarla (aggiungere barba, cambiare colore, ecc.), imposta is_edit a TRUE.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: { type: Type.STRING, description: 'La descrizione del contesto o della scena. Per modifiche, descrivi come deve apparire l\'immagine finale.' },
      is_selfie: { type: Type.BOOLEAN, description: 'TRUE se è una foto dell\'assistente, FALSE se oggetto generico o modifica di foto utente.' },
      is_uncensored: { type: Type.BOOLEAN, description: 'TRUE se l\'utente chiede esplicitamente contenuti senza censure/hot/sexy/intimi. FALSE per contenuti normali.' },
      is_edit: { type: Type.BOOLEAN, description: 'TRUE se l\'utente vuole MODIFICARE una foto che ha appena caricato (es. "aggiungi barba", "cambia colore capelli", "rendimi più giovane"). FALSE per generazioni da zero.' }
    },
    required: ['prompt'],
  },
};

const sendEmailTool: FunctionDeclaration = {
  name: 'send_email',
  description: 'Invia una email. PRIMA di usare questo strumento, DEVI chiedere all\'utente: 1) indirizzo email del destinatario, 2) oggetto della mail, 3) contenuto del messaggio. Solo quando hai tutte le informazioni, usa lo strumento.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      recipient: { type: Type.STRING, description: 'Indirizzo email del destinatario' },
      subject: { type: Type.STRING, description: 'Oggetto della email' },
      body: { type: Type.STRING, description: 'Corpo del messaggio' },
    },
    required: ['recipient', 'subject', 'body'],
  },
};

const sendWhatsappTool: FunctionDeclaration = {
  name: 'send_whatsapp',
  description: 'Invia un messaggio WhatsApp. PRIMA di usare questo strumento, DEVI chiedere all\'utente: 1) numero di telefono del destinatario (con prefisso internazionale, es. +39), 2) testo del messaggio. Solo quando hai tutte le informazioni, usa lo strumento.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      phoneNumber: { type: Type.STRING, description: 'Numero di telefono con prefisso internazionale (es. +393331234567)' },
      text: { type: Type.STRING, description: 'Testo del messaggio' },
    },
    required: ['phoneNumber', 'text'],
  },
};

const sendTelegramTool: FunctionDeclaration = {
  name: 'send_telegram',
  description: 'Invia un messaggio Telegram. PRIMA di usare questo strumento, DEVI chiedere all\'utente: 1) username Telegram del destinatario (senza @) OPPURE numero di telefono, 2) testo del messaggio. Solo quando hai tutte le informazioni, usa lo strumento.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      recipient: { type: Type.STRING, description: 'Username Telegram (senza @) o numero di telefono' },
      text: { type: Type.STRING, description: 'Testo del messaggio' },
    },
    required: ['recipient', 'text'],
  },
};

const makeCallTool: FunctionDeclaration = {
  name: 'make_call',
  description: 'Prepara una chiamata. PRIMA di usare questo strumento, chiedi: 1) Chi chiamare (Nome e Numero/Username), 2) Con quale app (Telefono classico, WhatsApp o Telegram).',
  parameters: {
    type: Type.OBJECT,
    properties: {
      recipient: { type: Type.STRING, description: 'Numero di telefono (per Tel/WhatsApp) o Username (per Telegram)' },
      app: { type: Type.STRING, description: 'L\'applicazione da usare', enum: ['phone', 'whatsapp', 'telegram'] },
      name: { type: Type.STRING, description: 'Il nome della persona da chiamare (per l\'etichetta)' }
    },
    required: ['recipient', 'app'],
  },
};

const getCalendarEventsTool: FunctionDeclaration = {
  name: 'get_calendar_events',
  description: 'Legge gli eventi dal calendario Google dell\'utente. Usa questo strumento quando l\'utente chiede di ricordargli gli appuntamenti, eventi, impegni o cosa ha in agenda. Puoi specificare quanti giorni nel futuro guardare.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      days_ahead: { type: Type.NUMBER, description: 'Numero di giorni nel futuro da controllare (default 7)' },
    },
    required: [],
  },
};

const createCalendarEventTool: FunctionDeclaration = {
  name: 'create_calendar_event',
  description: 'Crea un nuovo evento nel calendario Google dell\'utente. PRIMA di usare questo strumento, DEVI chiedere all\'utente: 1) titolo dell\'evento, 2) data e ora di inizio, 3) durata o ora di fine (opzionale, default 1 ora). Usa questo quando l\'utente vuole aggiungere/inserire/creare un appuntamento, evento, promemoria o impegno nel calendario.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      title: { type: Type.STRING, description: 'Titolo/nome dell\'evento' },
      start_datetime: { type: Type.STRING, description: 'Data e ora di inizio in formato ISO 8601 (es: 2025-01-15T14:00:00). Per eventi tutto il giorno usa solo la data (es: 2025-01-15)' },
      end_datetime: { type: Type.STRING, description: 'Data e ora di fine in formato ISO 8601. Se non specificato, l\'evento dura 1 ora' },
      description: { type: Type.STRING, description: 'Descrizione opzionale dell\'evento' },
      location: { type: Type.STRING, description: 'Luogo opzionale dell\'evento' },
    },
    required: ['title', 'start_datetime'],
  },
};

const allTools: Tool[] = [
  { 
    functionDeclarations: [
      generateImageTool, 
      sendEmailTool, 
      sendWhatsappTool, 
      sendTelegramTool,
      makeCallTool, 
      getCalendarEventsTool, 
      createCalendarEventTool
    ] 
  },
  { googleSearch: {} } 
];

let GOOGLE_CLIENT_ID = '';
try {
  // @ts-ignore
  GOOGLE_CLIENT_ID = (import.meta.env?.VITE_GOOGLE_CLIENT_ID || '').trim();
} catch(e) {}
if (!GOOGLE_CLIENT_ID) {
  try {
     // @ts-ignore
     GOOGLE_CLIENT_ID = (process.env?.VITE_GOOGLE_CLIENT_ID || '').trim();
  } catch(e) {}
}

const AppLogo = ({ size = 48, className = "" }: { size?: number, className?: string }) => {
  const [imgError, setImgError] = useState(false);

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <div className="relative h-full w-full bg-white/90 border border-white/50 rounded-[1rem] flex items-center justify-center overflow-hidden shadow-lg shadow-purple-200/40 backdrop-blur-sm">
        {!imgError ? (
           <img 
             src="/logo.png" 
             alt="Logo Ti Ascolto" 
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
  const [currentUser, setCurrentUser] = useState<FirebaseUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // --- RUBRICA STATE ---
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [showContactsModal, setShowContactsModal] = useState(false);
  const [newContactName, setNewContactName] = useState('');
  const [newContactPhone, setNewContactPhone] = useState('');
  const [newContactApp, setNewContactApp] = useState<'phone' | 'whatsapp' | 'telegram'>('phone');

  // Carica rubrica all'avvio
  useEffect(() => {
    const savedContacts = localStorage.getItem('ti_ascolto_contacts');
    if (savedContacts) {
      try { setContacts(JSON.parse(savedContacts)); } catch(e) {}
    }
  }, []);

  // Salva rubrica quando cambia
  useEffect(() => {
    localStorage.setItem('ti_ascolto_contacts', JSON.stringify(contacts));
  }, [contacts]);

  const addContact = () => {
    if (!newContactName || !newContactPhone) return;
    const newContact: Contact = {
      id: Date.now().toString(),
      name: newContactName,
      phone: newContactPhone,
      app: newContactApp
    };
    setContacts([...contacts, newContact]);
    setNewContactName('');
    setNewContactPhone('');
  };

  const deleteContact = (id: string) => {
    setContacts(contacts.filter(c => c.id !== id));
  };

  const handleImportContacts = async () => {
    // Verifica supporto Contact Picker API
    if ('contacts' in navigator && 'ContactsManager' in window) {
      try {
        const props = ['name', 'tel'];
        const opts = { multiple: true };
        
        // @ts-ignore
        const selectedContacts = await navigator.contacts.select(props, opts);
        
        if (selectedContacts.length > 0) {
          const newContacts: Contact[] = selectedContacts.map((c: any) => ({
            id: Date.now().toString() + Math.random().toString(),
            name: c.name[0], // Primo nome disponibile
            phone: c.tel[0], // Primo numero disponibile
            app: 'phone' // Default telefono
          }));
          
          setContacts(prev => [...prev, ...newContacts]);
          // alert(`Importati ${newContacts.length} contatti!`);
        }
      } catch (ex) {
        console.error("Errore importazione o annullato:", ex);
      }
    } else {
      alert("Il tuo dispositivo non supporta l'importazione automatica (Contact Picker API non disponibile).");
    }
  };

  const handleLogout = async () => {
    if (window.confirm('Vuoi effettuare il logout?')) {
      try {
        await signOut(auth);
        localStorage.removeItem('ti_ascolto_config');
        localStorage.removeItem('ti_ascolto_avatar');
        localStorage.removeItem('ti_ascolto_configured');
        localStorage.removeItem('ti_ascolto_chat_history');
        setIsConfigured(false);
        setAvatarUrl(null);
        setTranscripts([]);
      } catch (error) {
        console.error('Logout error:', error);
      }
    }
  };

  const [config, setConfig] = useState<AssistantConfig>({
    userName: '',
    gender: 'Donna',
    age: '25',
    hairColor: 'Castani',
    eyeColor: 'Verdi',
    skinTone: 'Chiara',
    bodyType: 'Normale',
    physicalTraits: '',
    personality: '',
    temperament: 'Calmo/a',
    sociality: 'Empatico',
    mood: 'Ottimista',
    commStyle: 'Buon ascoltatore',
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
  
  useEffect(() => {
    const savedConfig = localStorage.getItem('ti_ascolto_config');
    const savedAvatar = localStorage.getItem('ti_ascolto_avatar');
    const savedIsConfigured = localStorage.getItem('ti_ascolto_configured');
    
    if (savedIsConfigured === 'true' && savedConfig) {
      try {
        const parsedConfig = JSON.parse(savedConfig);
        setConfig(parsedConfig);
        if (savedAvatar) {
          setAvatarUrl(savedAvatar);
        }
        setIsConfigured(true);
      } catch (e) {
        console.warn('Errore caricamento configurazione:', e);
      }
    }
  }, []);
  
  const saveConfigToStorage = (configData: AssistantConfig, avatar: string | null) => {
    try {
      localStorage.setItem('ti_ascolto_config', JSON.stringify(configData));
      localStorage.setItem('ti_ascolto_configured', 'true');
      if (avatar) {
        localStorage.setItem('ti_ascolto_avatar', avatar);
      }
    } catch (e) {
      console.warn('Errore salvataggio configurazione:', e);
    }
  };
  
  const resetConfiguration = () => {
    localStorage.removeItem('ti_ascolto_config');
    localStorage.removeItem('ti_ascolto_avatar');
    localStorage.removeItem('ti_ascolto_configured');
    localStorage.removeItem('ti_ascolto_chat_history');
    setIsConfigured(false);
    setAvatarUrl(null);
    setTranscripts([]);
    disconnect();
  };
  
  const isFormComplete = config.userName.trim() !== '' && 
                         config.gender !== '' && 
                         config.age !== '' && 
                         config.hairColor !== '' && 
                         config.eyeColor !== '' && 
                         config.skinTone !== '' &&
                         config.bodyType !== '' &&
                         config.temperament !== '' &&
                         config.sociality !== '';
  
  const buildPersonality = () => {
    return `${config.temperament}, ${config.sociality}, ${config.mood}, ${config.commStyle}`;
  };

  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(false);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [audioVolume, setAudioVolume] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzingPhoto, setIsAnalyzingPhoto] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [googleCalendarToken, setGoogleCalendarToken] = useState<string | null>(null);

  useEffect(() => {
    const savedHistory = localStorage.getItem('ti_ascolto_chat_history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        setTranscripts(parsed);
      } catch (e) {
        console.error("Errore caricamento memoria:", e);
      }
    }
  }, []);

  useEffect(() => {
    if (transcripts.length > 0) {
      const historyToSave = transcripts.slice(-50); 
      localStorage.setItem('ti_ascolto_chat_history', JSON.stringify(historyToSave));
    }
  }, [transcripts]);

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
  const lastUserImageRef = useRef<string | null>(null);
  const lastUserImageAnalysisRef = useRef<string>(""); 
  const wakeLockRef = useRef<any>(null);
  const lastUploadedImageRef = useRef<string | null>(null);

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
    } else {
      console.warn("API Key mancante.");
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
        lastUserImageRef.current = base64Data; 
        const userImageId = Date.now().toString();
        addTranscript({ id: userImageId, sender: 'user', type: 'image', image: base64Data, isComplete: true });
        try {
          const imageAnalysisPrompt = `Analizza questa immagine in modo TECNICO per una rigenerazione AI (Image-to-Image). Descrivi: soggetto principale, abbigliamento, colori esatti, sfondo, illuminazione, stile fotografico. Sii dettagliatissimo.`;
          const response = await aiRef.current!.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: [{ role: 'user', parts: [{ text: imageAnalysisPrompt }, { inlineData: { mimeType: file.type, data: base64Data.split(',')[1] } }] }]
          });
          const analysis = response.text || "Immagine utente";
          lastUserImageAnalysisRef.current = analysis;
          if (isConnected && sessionPromiseRef.current) {
            const session = await sessionPromiseRef.current;
            session.sendClientContent({ 
              turns: [{ role: 'user', parts: [{ text: `[SYSTEM: L'utente ha caricato una foto. Analisi salvata: "${analysis}". SE chiede modifiche a QUESTA foto, usa 'generate_image' (is_selfie=false). SE chiede un selfie tuo, usa 'generate_image' (is_selfie=true).]` }] }] 
            });
          } else {
             addTranscript({ sender: 'model', type: 'text', text: "Ho visto la tua foto. Vuoi che la modifichi o ne vuoi una mia?", isComplete: true });
          }
        } catch (err) { console.error('Errore analisi', err); }
        setIsAnalyzingPhoto(false);
      };
      reader.readAsDataURL(file);
    } catch (err) { console.error(err); setIsAnalyzingPhoto(false); }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleConfigSubmit = async () => {
    if (!aiRef.current) { setError("API Key mancante."); return; }
    setIsGeneratingProfile(true);
    setError(null);
    const personalityString = buildPersonality();
    try {
        const hasManualName = config.name && config.name.trim().length > 0;
        setLoadingStep(hasManualName ? `Sto definendo la personalità di ${config.name}...` : 'Sto creando il tuo confidente ideale...');
        const basePrompt = `Crea un profilo per un COMPAGNO UMANO: Genere ${config.gender}, Età ${config.age}, Capelli ${config.hairColor}, Occhi ${config.eyeColor}, Pelle ${config.skinTone}, Corporatura ${config.bodyType || 'Normale'}, Caratteristiche fisiche: ${config.physicalTraits}, Personalità ${personalityString}.`;
        const nameInstruction = hasManualName ? `Il nome è "${config.name}".` : `Inventa un nome italiano creativo.`;
        const profilePrompt = `${basePrompt} ${nameInstruction} Rispondi JSON: {name, biography, visualPrompt}. IMPORTANTE: La biography DEVE essere scritta in ITALIANO.`;
        const textResponse = await aiRef.current.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: profilePrompt,
            config: { responseMimeType: 'application/json', responseSchema: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, biography: { type: Type.STRING }, visualPrompt: { type: Type.STRING } }, required: ['name', 'biography', 'visualPrompt'] } }
        });
        const profileData = JSON.parse(textResponse.text || '{}');
        setConfig(prev => ({ ...prev, name: profileData.name, biography: profileData.biography, visualPrompt: profileData.visualPrompt, personality: personalityString }));
        setLoadingStep(`Sto scattando una foto a ${profileData.name}...`);
        
        let foundUrl: string | null = null;
        const bodyTypeMap: {[key: string]: string} = { 'Minuta': 'petite', 'Normale': 'normal', 'Sportiva': 'athletic', 'Formoso/a': 'curvy', 'Taglia comoda': 'plus-size' };
        const bodyTypeEn = bodyTypeMap[config.bodyType] || 'normal';
        try {
            const imagePrompt = `Medium shot from hips up (American shot), visible waist and stomach, camera distance 3 meters. The subject is a friendly ${config.gender === 'Donna' ? 'woman' : config.gender === 'Uomo' ? 'man' : 'person'}, ${config.age} years old, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${bodyTypeEn} build. Wearing casual-elegant clothes suitable for a full torso shot. 8k resolution, photorealistic, soft studio lighting. ${profileData.visualPrompt}`;            
            const imageResponse = await aiRef.current.models.generateImages({ model: IMAGE_MODEL_NAME, prompt: imagePrompt, config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: '3:4' } });
            if (imageResponse.generatedImages?.[0]?.image?.imageBytes) {
                foundUrl = `data:image/jpeg;base64,${imageResponse.generatedImages[0].image.imageBytes}`;
            }
        } catch (imgError) {}
        setAvatarUrl(foundUrl);
        const updatedConfig = { ...config, name: profileData.name, biography: profileData.biography, visualPrompt: profileData.visualPrompt, personality: personalityString };
        saveConfigToStorage(updatedConfig, foundUrl);
        setIsConfigured(true);
    } catch (e: any) { setError("Errore creazione: " + e.message); } finally { setIsGeneratingProfile(false); setLoadingStep(''); }
  };

const handleImageGeneration = async (prompt: string, isSelfie: boolean = false, isUncensored: boolean = false): Promise<string | null> => {
    if (!aiRef.current) return null;
    if (isSelfie && isUncensored) {
        if (!avatarUrl) return "Non ho ancora una mia foto base per fare modifiche hot.";
        try {
            addTranscript({ sender: 'model', type: 'text', text: `🌶️ Ok, chiudo la porta...`, isComplete: true });
            // @ts-ignore
            const falKey = (import.meta.env?.VITE_FAL_KEY || process.env?.VITE_FAL_KEY || '').trim();
            if (!falKey) return "Manca la chiave VITE_FAL_KEY.";
            const response = await fetch('https://fal.run/fal-ai/bytedance/seedream/v4/edit', {
                method: 'POST', headers: { 'Authorization': `Key ${falKey}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt + ", uncensored, nsfw, highly detailed", input_image_urls: [avatarUrl], image_size: "portrait_4_3", sync_mode: true })
            });
            const data = await response.json();
            const imgUrl = data.images?.[0]?.url || data.image?.url;
            if (imgUrl) { addTranscript({ sender: 'model', type: 'image', image: imgUrl, isComplete: true }); return "Ecco qui (versione senza censure)."; }
        } catch (e) { return "Errore nella generazione speciale."; }
        return "Errore sconosciuto Strada 3.";
    }
    let basePrompt = "";
    if (isSelfie) {
        const bodyMap: any = { 'Minuta': 'petite', 'Normale': 'normal', 'Sportiva': 'athletic', 'Formoso/a': 'curvy' };
        const bodyEn = bodyMap[config.bodyType] || 'normal';
        basePrompt = `Photorealistic portrait of a ${config.age} year old ${config.gender}, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${bodyEn} build. ${config.physicalTraits || ''}.`;
        addTranscript({ sender: 'model', type: 'text', text: `📸 Scatto la foto...`, isComplete: true });
    } else {
        if (!lastUserImageAnalysisRef.current) return "Non posso modificare la foto.";
        basePrompt = `Recreate this specific image: ${lastUserImageAnalysisRef.current}.`;
        addTranscript({ sender: 'model', type: 'text', text: `🎨 Modifico la tua foto...`, isComplete: true });
    }
    const finalImagenPrompt = `${basePrompt} MODIFICATIONS: ${prompt}. Style: 8k, photorealistic, raw photo, natural lighting.`;
    try {
        const response = await aiRef.current.models.generateImages({ model: IMAGE_MODEL_NAME, prompt: finalImagenPrompt, config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: '3:4' } });
        const img = response.generatedImages?.[0];
        if (img?.image?.imageBytes) {
            const url = `data:image/jpeg;base64,${img.image.imageBytes}`;
            addTranscript({ sender: 'model', type: 'image', image: url, isComplete: true });
            return "Fatto.";
        }
        return "L'immagine è stata bloccata dai filtri di sicurezza.";
    } catch (e) { return "Errore tecnico durante la generazione."; }
  };
  const handleSendEmail = (recipient: string, subject: string, body: string) => {
    addTranscript({ sender: 'model', type: 'action', text: `📧 Email pronta per: ${recipient}`, isComplete: true, actionUrl: `mailto:${recipient}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`, actionLabel: 'Invia Email', actionIcon: 'mail' });
    return "SUCCESS";
  };
  const handleSendWhatsapp = (phoneNumber: string, text: string) => {
    addTranscript({ sender: 'model', type: 'action', text: `💬 WhatsApp pronto per: ${phoneNumber}`, isComplete: true, actionUrl: `https://wa.me/${phoneNumber.replace(/\D/g, '')}?text=${encodeURIComponent(text)}`, actionLabel: 'Invia WhatsApp', actionIcon: 'message-circle' });
    return "SUCCESS";
  };
  const handleSendTelegram = (recipient: string, text: string) => {
    const isPhoneNumber = /^\+?\d+$/.test(recipient.replace(/\s/g, ''));
    const telegramUrl = isPhoneNumber ? `https://t.me/+${recipient.replace(/\D/g, '')}?text=${encodeURIComponent(text)}` : `https://t.me/${recipient.replace('@', '')}?text=${encodeURIComponent(text)}`;
    addTranscript({ sender: 'model', type: 'action', text: `✈️ Telegram pronto per: ${recipient}`, isComplete: true, actionUrl: telegramUrl, actionLabel: 'Invia Telegram', actionIcon: 'send' });
    return "SUCCESS";
  };
  const handleMakeCall = (recipient: string, app: string, name?: string) => {
    let actionUrl = '', label = '', iconName = 'phone';
    const cleanNum = recipient.replace(/\D/g, '');
    if (app.toLowerCase().includes('whatsapp')) {
      actionUrl = `https://wa.me/${cleanNum}`; label = `Apri WhatsApp per chiamare ${name || recipient}`; iconName = 'message-circle';
    } else if (app.toLowerCase().includes('telegram')) {
      const username = recipient.replace('@', '').replace('https://t.me/', '');
      actionUrl = `https://t.me/${username}`; label = `Apri Telegram per chiamare ${name || recipient}`; iconName = 'send';
    } else {
      actionUrl = `tel:+${cleanNum}`; label = `Chiama ${name || recipient} al cellulare`; iconName = 'phone';
    }
    addTranscript({ sender: 'model', type: 'action', text: `📞 Chiamata pronta con ${app}...`, isComplete: true, actionUrl: actionUrl, actionLabel: label, actionIcon: iconName });
    return "SUCCESS: Link di chiamata generato.";
  };

  const initGoogleCalendar = () => {
    if (!GOOGLE_CLIENT_ID) return;
    const redirectUri = window.location.origin;
    const scope = 'https://www.googleapis.com/auth/calendar';
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${encodeURIComponent(GOOGLE_CLIENT_ID)}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=token&scope=${encodeURIComponent(scope)}&prompt=consent`;
    const handleAuthMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;
      if (event.data && event.data.type === 'GOOGLE_AUTH_SUCCESS') {
        setGoogleCalendarToken(event.data.token);
        localStorage.setItem('google_calendar_token', event.data.token);
        window.removeEventListener('message', handleAuthMessage);
      }
    };
    window.addEventListener('message', handleAuthMessage);
    const popup = window.open(authUrl, 'google-auth', 'width=500,height=600');
    if (!popup) { setError("Il browser ha bloccato il popup."); window.removeEventListener('message', handleAuthMessage); }
  };

  useEffect(() => {
    const savedToken = localStorage.getItem('google_calendar_token');
    if (savedToken) setGoogleCalendarToken(savedToken);
    if (window.location.hash.includes('access_token')) {
      const accessToken = new URLSearchParams(window.location.hash.substring(1)).get('access_token');
      if (accessToken) {
        if (window.opener) { window.opener.postMessage({ type: 'GOOGLE_AUTH_SUCCESS', token: accessToken }, window.location.origin); window.close(); } 
        else { setGoogleCalendarToken(accessToken); localStorage.setItem('google_calendar_token', accessToken); window.history.replaceState({}, document.title, window.location.pathname); }
      }
    }
  }, []);

  const handleGetCalendarEvents = async (daysAhead: number = 7): Promise<string> => {
    if (!googleCalendarToken) return "Il calendario Google non è connesso.";
    try {
      const now = new Date(), futureDate = new Date(); futureDate.setDate(now.getDate() + daysAhead);
      const url = `https://www.googleapis.com/calendar/v3/calendars/primary/events?timeMin=${now.toISOString()}&timeMax=${futureDate.toISOString()}&singleEvents=true&orderBy=startTime`;
      const response = await fetch(url, { headers: { 'Authorization': `Bearer ${googleCalendarToken}` } });
      if (!response.ok) { if (response.status === 401) { setGoogleCalendarToken(null); localStorage.removeItem('google_calendar_token'); return "Token scaduto."; } return "Errore lettura calendario."; }
      const data = await response.json();
      const events = data.items || [];
      if (events.length === 0) return `NESSUN EVENTO TROVATO per i prossimi ${daysAhead} giorni.`;
      const eventList = events.map((event: any) => {
        const start = event.start?.dateTime || event.start?.date;
        const startDate = new Date(start);
        const dateStr = startDate.toLocaleDateString('it-IT', { weekday: 'long', day: 'numeric', month: 'long' });
        const timeStr = event.start?.dateTime ? startDate.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' }) : "Tutto il giorno";
        return `- GIORNO: ${dateStr} | ORA: ${timeStr} | TITOLO: ${event.summary || '(Senza titolo)'}`;
      }).join('\n');
      return `Ecco la lista ESATTA degli eventi:\n${eventList}`;
    } catch (e) { return "Errore imprevisto calendario."; }
  };

  const handleCreateCalendarEvent = async (title: string, startDatetime: string, endDatetime?: string, description?: string, location?: string): Promise<string> => {
    if (!googleCalendarToken) return "Il calendario Google non è connesso.";
    try {
      const isAllDay = !startDatetime.includes('T');
      let eventBody: any = { summary: title };
      if (description) eventBody.description = description; if (location) eventBody.location = location;
      if (isAllDay) {
        eventBody.start = { date: startDatetime };
        if (endDatetime) eventBody.end = { date: endDatetime };
        else { const nextDay = new Date(startDatetime); nextDay.setDate(nextDay.getDate() + 1); eventBody.end = { date: nextDay.toISOString().split('T')[0] }; }
      } else {
        let startISO = startDatetime; if (startDatetime.split('T')[1]?.split(':').length === 2 && !startDatetime.includes('Z')) startISO += ':00';
        eventBody.start = { dateTime: startISO, timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone };
        if (endDatetime) { let endISO = endDatetime; if (endDatetime.split('T')[1]?.split(':').length === 2 && !endDatetime.includes('Z')) endISO += ':00'; eventBody.end = { dateTime: endISO, timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone }; }
        else { const endTime = new Date(startISO); endTime.setHours(endTime.getHours() + 1); eventBody.end = { dateTime: endTime.toISOString(), timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone }; }
      }
      const response = await fetch('https://www.googleapis.com/calendar/v3/calendars/primary/events', { method: 'POST', headers: { 'Authorization': `Bearer ${googleCalendarToken}`, 'Content-Type': 'application/json' }, body: JSON.stringify(eventBody) });
      if (!response.ok) return "Errore creazione evento.";
      return "EVENTO CREATO CON SUCCESSO!";
    } catch (e) { return "Errore imprevisto creazione evento."; }
  };

  const disconnectGoogleCalendar = () => { setGoogleCalendarToken(null); localStorage.removeItem('google_calendar_token'); };

  const connect = async () => {
    if (!aiRef.current) { setError("Chiave API non trovata."); return; }
    setError(null);
    try {
      if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') { setError("HTTPS richiesto."); return; }
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      let stream: MediaStream;
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); } catch (micError: any) { setError("Errore microfono: " + micError.message); return; }
      try { if ('wakeLock' in navigator) { wakeLockRef.current = await (navigator as any).wakeLock.request('screen'); } } catch (e) {}
      
      const ageNum = parseInt(config.age) || 30;
      let selectedVoiceName = config.gender === 'Uomo' ? (ageNum < 35 ? 'Puck' : 'Fenrir') : (ageNum < 35 ? 'Aoede' : 'Kore');
      const rawSociality = config.sociality.split('/')[0]; 
      const personalityProfile = PERSONALITY_PROMPTS[rawSociality] || PERSONALITY_PROMPTS['Empatico'];
      
      const memoryContext = transcripts.filter(t => t.type === 'text').slice(-15).map(t => `${t.sender === 'user' ? (config.userName || 'Utente') : config.name}: ${t.text}`).join('\n');
      const memoryInstruction = memoryContext ? `\n\n=== MEMORIA RECENTE ===\n${memoryContext}\n` : "";

      // 1. Prepara la stringa della rubrica
      const contactsList = contacts.length > 0 
        ? contacts.map(c => `- ${c.name}: ${c.phone} (preferenza: ${c.app})`).join('\n')
        : "Nessun contatto salvato in rubrica.";

      const configLive = {
        model: LIVE_MODEL_NAME,
        generationConfig: { temperature: personalityProfile.temp },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoiceName } } },
          systemInstruction: `
--- MODULO PERSONALITÀ ATTIVO: ${rawSociality.toUpperCase()} ---
${personalityProfile.prompt}
-----------------------------------------------------------
${memoryInstruction}

=== RUBRICA CONTATTI DELL'UTENTE ===
Usa questi numeri quando l'utente ti chiede di chiamare qualcuno per nome.
${contactsList}
====================================

Sei ${config.name}, confidente di ${config.userName}. 
Bio: ${config.biography}.

RICERCA ONLINE E INFO LOCALI:
- Hai accesso allo strumento 'googleSearch'. Usalo per info locali, notizie e dati.

GESTIONE E MODIFICA IMMAGINI:
- Se l'utente carica una foto e chiede modifiche, usa 'generate_image' con is_edit=TRUE.
- Se chiede un tuo selfie, usa 'generate_image' con is_selfie=TRUE.

CHIAMATE E MESSAGGI:
- Se l'utente vuole CHIAMARE o TELEFONARE:
  1. Cerca il nome nella RUBRICA CONTATTI qui sopra.
  2. Se trovi il contatto, usa 'make_call' con il suo numero e app preferita.
  3. Se NON lo trovi, chiedi numero e app.
- Se l'utente vuole mandare MESSAGGI (Email/WhatsApp/Telegram): chiedi prima tutte le info necessarie.

CALENDARIO:
- Usa 'get_calendar_events' e 'create_calendar_event' secondo richiesta.

Parla sempre in italiano rispettando il Tono del Modulo Personalità.`,
          tools: allTools,
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
              let sum = 0; for(let i=0;i<inputData.length;i++) sum+=inputData[i]*inputData[i];
              if(Math.random()>0.8) setAudioVolume(Math.sqrt(sum/inputData.length)*5);
              if(isMutedRef.current) return;
              sessionPromiseRef.current?.then(session => session.sendRealtimeInput({ media: createBlob(inputData) })).catch(console.error);
            };
            source.connect(processor); processor.connect(ctx.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
             if (msg.toolCall) {
                for (const fc of msg.toolCall.functionCalls) {
                    let res = "OK";
                    if (fc.name === 'generate_image') res = await handleImageGeneration((fc.args as any).prompt, (fc.args as any).is_selfie, (fc.args as any).is_uncensored, (fc.args as any).is_edit) || "Err";
                    else if (fc.name === 'send_email') res = handleSendEmail((fc.args as any).recipient, (fc.args as any).subject, (fc.args as any).body);
                    else if (fc.name === 'send_whatsapp') res = handleSendWhatsapp((fc.args as any).phoneNumber, (fc.args as any).text);
                    else if (fc.name === 'send_telegram') res = handleSendTelegram((fc.args as any).recipient, (fc.args as any).text);
                    else if (fc.name === 'make_call') res = handleMakeCall((fc.args as any).recipient, (fc.args as any).app, (fc.args as any).name);
                    else if (fc.name === 'get_calendar_events') res = await handleGetCalendarEvents((fc.args as any).days_ahead || 7);
                    else if (fc.name === 'create_calendar_event') res = await handleCreateCalendarEvent((fc.args as any).title, (fc.args as any).start_datetime, (fc.args as any).end_datetime, (fc.args as any).description, (fc.args as any).location);
                    sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: [{ id: fc.id, name: fc.name, response: { result: res } }] }));
                }
             }
            if (msg.serverContent?.outputTranscription) { currentOutputTransRef.current += msg.serverContent.outputTranscription.text; addTranscript({ text: currentOutputTransRef.current, sender: 'model', type: 'text', isComplete: false }); }
            if (msg.serverContent?.inputTranscription) { currentInputTransRef.current += msg.serverContent.inputTranscription.text; addTranscript({ text: currentInputTransRef.current, sender: 'user', type: 'text', isComplete: false }); }
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
                const detuneFactor = Math.pow(2, (config.voicePitch || 0) / 1200);
                const effectiveSpeed = (config.voiceSpeed || 1.0) * detuneFactor;
                source.detune.value = config.voicePitch || 0;
                source.connect(ctx.destination);
                source.addEventListener('ended', () => { audioSourcesRef.current.delete(source); if(audioSourcesRef.current.size===0) setAudioVolume(0); });
                setAudioVolume(0.5);
                source.start(nextStartTimeRef.current);
                nextStartTimeRef.current += buffer.duration / effectiveSpeed;
                audioSourcesRef.current.add(source);
            }
            if (msg.serverContent?.interrupted) { audioSourcesRef.current.forEach(s => s.stop()); audioSourcesRef.current.clear(); nextStartTimeRef.current = 0; currentOutputTransRef.current = ''; }
          },
          onclose: () => setIsConnected(false),
          onerror: (e) => { console.error(e); setError(`Errore connessione: ${e.message}`); disconnect(); }
        }
      });
      sessionPromiseRef.current = sessionPromise;
    } catch (err: any) { setError(err.message); disconnect(); }
  };

  const disconnect = () => {
    sessionPromiseRef.current?.then(s => s.close()).catch(()=>{}); sessionPromiseRef.current = null;
    inputSourceRef.current?.disconnect(); processorRef.current?.disconnect();
    inputAudioContextRef.current?.close(); outputAudioContextRef.current?.close();
    if (wakeLockRef.current) { wakeLockRef.current.release().catch(() => {}); wakeLockRef.current = null; }
    setIsConnected(false); setAudioVolume(0);
  };
  const toggleMute = () => { setIsMuted(!isMuted); isMutedRef.current = !isMuted; };
  const transcriptRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight; }, [transcripts]);
  useEffect(() => {
    const handleVisibilityChange = async () => { if (document.visibilityState === 'visible' && isConnected && !wakeLockRef.current) { try { if ('wakeLock' in navigator) wakeLockRef.current = await (navigator as any).wakeLock.request('screen'); } catch (e) {} } };
    document.addEventListener('visibilitychange', handleVisibilityChange); return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [isConnected]);

  if (authLoading) return <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc' }}><Loader2 className="animate-spin" size={48} color="#9333ea"/></div>;
  if (!currentUser) return <AuthScreen onAuthSuccess={() => {}} />;
if (!isConfigured) {
    return (
        <div style={{ minHeight: '100vh', backgroundImage: "url('/background.png')", backgroundSize: 'cover', backgroundPosition: 'center', backgroundColor: '#f8fafc', position: 'relative', fontFamily: 'Outfit, sans-serif', color: '#1e293b' }}>
            <style>{`
              @keyframes heartPulse { 0%, 100% { transform: scale(1); box-shadow: 0 0 15px rgba(147, 51, 234, 0.6); } 50% { transform: scale(1.15); box-shadow: 0 0 30px rgba(147, 51, 234, 0.9); } }
              input[type="range"]::-webkit-slider-thumb { appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #9333ea; cursor: pointer; box-shadow: 0 2px 6px rgba(147, 51, 234, 0.4); }
              .animate-spin { animation: spin 1s linear infinite; } @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
              @media (max-width: 900px) { .config-container { flex-direction: column !important; padding: 20px !important; gap: 20px !important; } .config-left-column { flex: none !important; width: 100% !important; text-align: center !important; } .config-right-column { width: 100% !important; } .config-grid-4 { grid-template-columns: 1fr 1fr !important; } .personality-grid { grid-template-columns: 1fr !important; } }
            `}</style>
            <div className="config-container" style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px', position: 'relative', zIndex: 10, display: 'flex', gap: '60px', minHeight: '100vh', alignItems: 'center' }}>
                <div className="config-left-column" style={{ flex: '0 0 420px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
                        <div style={{ width: '80px', height: '80px', backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 24px rgba(147, 112, 219, 0.25)', overflow: 'hidden' }}>
                          <img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                        </div>
                        <div><div style={{ fontSize: '12px', fontWeight: 700, letterSpacing: '0.2em', color: '#64748b', textTransform: 'uppercase' }}>Progetto</div><div style={{ fontSize: '22px', fontWeight: 700, color: '#1e293b', letterSpacing: '-0.02em' }}>TI ASCOLTO</div></div>
                    </div>
                    <h1 style={{ fontSize: '56px', fontWeight: 700, color: '#0f172a', marginBottom: '24px', lineHeight: 1.1, letterSpacing: '-0.02em' }}>Ciao,<br/>Parliamo, ti va?</h1>
                    <p style={{ fontSize: '17px', color: '#475569', fontWeight: 500, lineHeight: 1.7, marginBottom: '32px' }}>Sono qualcuno che ti ascolta davvero. Configurami, dammi un volto e una voce.</p>
                </div>
                <div className="config-right-column" style={{ flex: 1, maxWidth: '650px' }}>
                    <div className="config-form" style={{ padding: '20px', maxHeight: '85vh', overflowY: 'auto' }}>
                        {error && (<div style={{ marginBottom: '24px', padding: '16px', backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '12px', color: '#dc2626', fontSize: '14px' }}><Info size={16} /> {error}</div>)}
                        <div style={{ marginBottom: '28px' }}><h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '12px' }}>Chi sei tu?</h3><input style={{ width: '100%', padding: '16px', borderRadius: '16px', border: '1px solid #e2e8f0', background: 'rgba(255,255,255,0.7)' }} placeholder="Il tuo nome" value={config.userName} onChange={(e) => setConfig({...config, userName: e.target.value})} /></div>
                        <div style={{ marginBottom: '28px' }}><h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '12px' }}>Il tuo Confidente</h3>
                            <div className="config-grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                                <select style={{ width: '100%', padding: '16px', borderRadius: '16px', border: '1px solid #e2e8f0', background: 'rgba(255,255,255,0.7)' }} value={config.gender} onChange={(e) => setConfig({...config, gender: e.target.value})}><option>Uomo</option><option>Donna</option><option>Non-binary</option></select>
                                <input type="number" style={{ width: '100%', padding: '16px', borderRadius: '16px', border: '1px solid #e2e8f0', background: 'rgba(255,255,255,0.7)' }} value={config.age} onChange={(e) => setConfig({...config, age: e.target.value})} />
                            </div>
                        </div>
                        <button onClick={isFormComplete && !isGeneratingProfile ? handleConfigSubmit : undefined} disabled={!isFormComplete || isGeneratingProfile} style={{ width: '100%', padding: '20px', borderRadius: '16px', backgroundColor: isFormComplete ? '#9333ea' : '#e2e8f0', color: 'white', fontWeight: 700, border: 'none', cursor: isFormComplete ? 'pointer' : 'not-allowed' }}>{isGeneratingProfile ? (loadingStep || 'Creazione...') : 'Crea il tuo Confidente'}</button>
                    </div>
                </div>
            </div>
        </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'row', height: '100vh', width: '100%', backgroundImage: "url('/background.png')", backgroundSize: 'cover', backgroundPosition: 'center', backgroundColor: '#f8fafc', position: 'relative', fontFamily: 'Outfit, sans-serif', color: '#1e293b', overflow: 'hidden' }}>
      <style>{`
        @keyframes glowPulseGreen { 0%, 100% { box-shadow: 0 0 15px 5px rgba(34, 197, 94, 0.6); } 50% { box-shadow: 0 0 25px 10px rgba(34, 197, 94, 0.9); } }
        @keyframes glowPulseOrange { 0%, 100% { box-shadow: 0 0 10px 3px rgba(249, 115, 22, 0.5); } 50% { box-shadow: 0 0 18px 6px rgba(249, 115, 22, 0.8); } }
        @media (max-width: 768px) { .chat-sidebar { display: none !important; } .chat-main { width: 100% !important; } .desktop-visualizer { display: none !important; } .mobile-header-container { display: flex !important; } .mobile-calendar-container { display: none !important; } }
        @media (min-width: 769px) { .mobile-header-container { display: none !important; } .desktop-visualizer { display: flex !important; } }
      `}</style>

      <aside className="chat-sidebar" style={{ width: '380px', minWidth: '300px', maxWidth: '450px', backgroundColor: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(12px)', borderRight: '1px solid rgba(226,232,240,0.6)', display: 'flex', flexDirection: 'column', padding: '20px', overflowY: 'auto', zIndex: 10 }}>
        <div onClick={() => { if(window.confirm('Tornare al menu?')) resetConfiguration(); }} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', cursor: 'pointer' }}>
          <div style={{ width: '40px', height: '40px', backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}><img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} /></div>
          <div><div style={{ fontSize: '8px', fontWeight: 700, letterSpacing: '0.15em', color: '#64748b' }}>PROGETTO</div><div style={{ fontSize: '16px', fontWeight: 700 }}>Ti Ascolto</div></div>
        </div>
        <h2 style={{ fontSize: '22px', fontWeight: 700, marginBottom: '2px' }}>{config.name}</h2>
        <div style={{ width: '100%', paddingBottom: '133%', position: 'relative', borderRadius: '12px', overflow: 'hidden', backgroundColor: '#f1f5f9', marginBottom: '12px' }}>
          {avatarUrl ? <img src={avatarUrl} alt="Avatar" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }} /> : <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><User size={40} color="#cbd5e1" /></div>}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ marginTop: 'auto' }}>
          {!isConnected ? ( <button onClick={connect} style={{ width: '100%', padding: '14px', backgroundColor: '#0f172a', color: 'white', borderRadius: '12px', fontWeight: 700, border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}><Mic size={18} /> INIZIA A PARLARE</button> ) : (
            <div style={{ display: 'flex', gap: '8px' }}>
              <button onClick={toggleMute} style={{ flex: 1, padding: '12px', backgroundColor: isMuted ? '#fef2f2' : 'white', color: isMuted ? '#ef4444' : '#475569', borderRadius: '10px', border: isMuted ? '1px solid #fecaca' : '1px solid #e2e8f0', display: 'flex', justifyContent: 'center' }}>{isMuted ? <MicOff size={18} /> : <Mic size={18} />}</button>
              <button onClick={disconnect} style={{ flex: 2, padding: '12px', backgroundColor: '#fef2f2', color: '#ef4444', borderRadius: '10px', border: '1px solid #fecaca', fontWeight: 700, display: 'flex', justifyContent: 'center', gap: '6px' }}><PhoneOff size={16} /> Termina</button>
            </div>
          )}
          
          <button onClick={() => setShowContactsModal(true)} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', padding: '10px', marginTop: '8px', backgroundColor: 'transparent', color: '#4f46e5', borderRadius: '10px', border: '1px dashed #818cf8', cursor: 'pointer', fontSize: '12px', fontWeight: 500 }}><BookUser size={14} /> Gestisci Rubrica</button>

          <div style={{ marginTop: '12px' }}>
              {googleCalendarToken ? ( <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', backgroundColor: '#f0fdf4', borderRadius: '10px', border: '1px solid #bbf7d0', fontSize: '11px', color: '#16a34a' }}><span>Calendario connesso</span><button onClick={disconnectGoogleCalendar} style={{ border: 'none', background: 'transparent', textDecoration: 'underline', cursor: 'pointer', color: '#64748b' }}>Disconnetti</button></div> ) : ( <button onClick={initGoogleCalendar} style={{ width: '100%', padding: '10px', backgroundColor: 'white', color: '#16a34a', borderRadius: '12px', border: '1px solid #bbf7d0', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', fontSize: '12px', fontWeight: 600 }}><Calendar size={16} /> Connetti Calendar</button> )}
          </div>
          <button onClick={handleLogout} style={{ width: '100%', padding: '10px', marginTop: '8px', backgroundColor: 'transparent', color: '#ef4444', borderRadius: '10px', border: '1px dashed #fca5a5', cursor: 'pointer', fontSize: '12px' }}>Esci</button>
        </div>
      </aside>

      <main className="chat-main" style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', zIndex: 10, height: '100%', backgroundColor: 'rgba(255,255,255,0.3)' }}>
        {error && <div style={{ position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)', zIndex: 50, padding: '12px 24px', backgroundColor: '#ef4444', color: 'white', borderRadius: '12px', boxShadow: '0 8px 24px rgba(239, 68, 68, 0.3)', fontSize: '14px' }}><Info size={18} /> {error}</div>}
        
        <div className="mobile-header-container" style={{ padding: '12px', borderBottom: '1px solid #e2e8f0', backgroundColor: 'rgba(255,255,255,0.98)', display: 'none' }}>
           <div style={{ display: 'flex', gap: '12px' }}>
             <div onClick={() => { if(window.confirm('Tornare al menu?')) resetConfiguration(); }} style={{ width: '80px', height: '80px', borderRadius: '8px', overflow: 'hidden', flexShrink: 0, boxShadow: !isConnected ? 'none' : '0 0 10px rgba(34,197,94,0.5)' }}>{avatarUrl ? <img src={avatarUrl} style={{ width: '100%', height: '100%', objectFit: 'cover' }}/> : <div style={{ background: '#f1f5f9', width: '100%', height: '100%' }}/>}</div>
             <div style={{ flex: 1 }}>
               <h2 style={{ fontSize: '18px', fontWeight: 700 }}>{config.name}</h2>
               <p style={{ fontSize: '11px', color: '#64748b' }}>Confidente di {config.userName}</p>
             </div>
           </div>
        </div>

        <div className="desktop-visualizer" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '16px', padding: '10px', borderBottom: '1px solid rgba(226,232,240,0.4)', backgroundColor: 'rgba(255,255,255,0.4)' }}>
          <div style={{ position: 'relative', transform: 'scale(0.5)' }}>
            <AudioVisualizer isPlaying={isConnected} volume={audioVolume} />
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{isConnected ? (isMuted ? <MicOff /> : <Mic color="#9333ea"/>) : <PhoneOff color="#e2e8f0"/>}</div>
          </div>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' }}>{isConnected ? `In conversazione con ${config.name}` : "Premi 'Inizia a parlare'"}</p>
        </div>

        <div ref={transcriptRef} className="chat-transcript" style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {transcripts.length === 0 && <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#cbd5e1', gap: '8px' }}><Sparkles size={32} /><span>Inizia la conversazione...</span></div>}
          {transcripts.map((t) => (
            <div key={t.id} style={{ display: 'flex', justifyContent: t.sender === 'user' ? 'flex-end' : 'flex-start' }}>
              <div style={{ maxWidth: '70%', borderRadius: '16px', padding: t.type === 'action' ? '0' : '16px 20px', fontSize: '14px', lineHeight: 1.6, backgroundColor: t.sender === 'user' ? '#0f172a' : t.type === 'action' ? 'transparent' : 'white', color: t.sender === 'user' ? 'white' : '#334155', boxShadow: t.type === 'action' ? 'none' : '0 2px 12px rgba(0,0,0,0.08)' }}>
                {t.type !== 'action' && <div style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', marginBottom: '6px', color: t.sender === 'user' ? '#94a3b8' : '#9333ea' }}>{t.sender === 'user' ? 'Tu' : config.name}</div>}
                {t.type === 'text' && <div>{t.text}</div>}
                {t.type === 'image' && t.image && (<div style={{ marginTop: '8px', borderRadius: '12px', overflow: 'hidden', position: 'relative' }}><img src={t.image} style={{ width: '100%', maxHeight: '400px', objectFit: 'cover' }} /><button onClick={() => t.image && downloadImage(t.image, 'foto.png')} style={{ position: 'absolute', bottom: '8px', right: '8px', padding: '8px', background: 'white', borderRadius: '50%', border: 'none' }}><Download size={14} /></button></div>)}
                {t.type === 'action' && t.actionUrl && (
                  <button onClick={() => { if(t.actionUrl?.startsWith('mailto:')) window.location.href = t.actionUrl; else window.open(t.actionUrl, '_blank'); }} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px', borderRadius: '12px', fontWeight: 700, color: 'white', border: 'none', cursor: 'pointer', width: '100%', background: t.actionIcon === 'phone' ? 'linear-gradient(135deg, #f59e0b, #d97706)' : 'linear-gradient(135deg, #0088cc, #00a0dc)' }}>
                    <div style={{ padding: '8px', background: 'rgba(255,255,255,0.2)', borderRadius: '50%' }}>{t.actionIcon === 'phone' ? <Phone size={20} /> : <MessageCircle size={20} />}</div>
                    <div style={{ display: 'flex', flexDirection: 'column', textAlign: 'left' }}><span style={{ fontSize: '15px' }}>{t.actionLabel}</span><span style={{ fontSize: '10px', opacity: 0.8 }}>Tocca per aprire</span></div>
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="chat-footer" style={{ borderTop: '1px solid rgba(226,232,240,0.5)', backgroundColor: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(8px)', flexShrink: 0 }}>
          <div className="mobile-calendar-container" style={{ padding: '0 16px' }}>
             <button onClick={() => setShowContactsModal(true)} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', padding: '10px 14px', marginTop: '8px', backgroundColor: '#eef2ff', color: '#4f46e5', borderRadius: '10px', border: '1px solid #c7d2fe', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}><BookUser size={14} /> Rubrica Contatti</button>
          </div>
          <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <input type="file" ref={fileInputRef} accept="image/*" onChange={handleUserPhotoUpload} style={{ display: 'none' }} />
            <button onClick={() => fileInputRef.current?.click()} disabled={isAnalyzingPhoto} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '12px 16px', backgroundColor: 'white', color: isAnalyzingPhoto ? '#94a3b8' : '#64748b', border: '1px solid #e2e8f0', borderRadius: '10px', fontSize: '12px', fontWeight: 600, cursor: 'pointer' }}>{isAnalyzingPhoto ? <Loader2 size={14} className="animate-spin" /> : <Camera size={14} />}<span>Foto</span></button>
            {!isConnected ? ( <button onClick={connect} style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', padding: '12px 20px', backgroundColor: '#0f172a', color: 'white', borderRadius: '10px', fontWeight: 700, fontSize: '13px', border: 'none' }}><Mic size={16} /> INIZIA A PARLARE</button> ) : (
              <div style={{ flex: 1, display: 'flex', gap: '8px' }}>
                <button onClick={toggleMute} style={{ flex: 1, padding: '12px', backgroundColor: isMuted ? '#fef2f2' : 'white', color: isMuted ? '#ef4444' : '#22c55e', borderRadius: '10px', border: isMuted ? '1px solid #fecaca' : '1px solid #bbf7d0', display: 'flex', justifyContent: 'center' }}>{isMuted ? <MicOff size={18} /> : <Mic size={18} />}</button>
                <button onClick={disconnect} style={{ flex: 2, padding: '12px', backgroundColor: '#fef2f2', color: '#ef4444', borderRadius: '10px', fontWeight: 700, border: '1px solid #fecaca', display: 'flex', justifyContent: 'center' }}><PhoneOff size={16} /> Termina</button>
              </div>
            )}
          </div>
          <div style={{ padding: '10px', textAlign: 'center', fontSize: '8px', color: '#94a3b8' }}>© Effetre Properties IA Division 2025</div>
        </div>
      </main>

      {/* --- MODALE RUBRICA --- */}
      {showContactsModal && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
          <div style={{ backgroundColor: 'white', borderRadius: '24px', width: '100%', maxWidth: '400px', boxShadow: '0 20px 50px rgba(0,0,0,0.2)', overflow: 'hidden', display: 'flex', flexDirection: 'column', maxHeight: '80vh' }}>
            <div style={{ padding: '20px', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#f8fafc' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><BookUser className="text-purple-600" size={24} /><h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#1e293b' }}>Rubrica</h3></div>
              <button onClick={() => setShowContactsModal(false)} style={{ border: 'none', background: 'transparent', cursor: 'pointer' }}><X size={24} className="text-slate-400" /></button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {contacts.length === 0 ? (<p style={{ textAlign: 'center', color: '#94a3b8', fontSize: '14px', marginTop: '20px' }}>La rubrica è vuota.</p>) : (
                contacts.map(c => (
                  <div key={c.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', backgroundColor: '#f1f5f9', borderRadius: '12px' }}>
                    <div><div style={{ fontWeight: 700, color: '#1e293b' }}>{c.name}</div><div style={{ fontSize: '12px', color: '#64748b' }}>{c.phone} • {c.app}</div></div>
                    <button onClick={() => deleteContact(c.id)} style={{ color: '#ef4444', background: 'transparent', border: 'none', cursor: 'pointer' }}><Trash2 size={18} /></button>
                  </div>
                ))
              )}
            </div>
            <div style={{ padding: '20px', borderTop: '1px solid #e2e8f0', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {'contacts' in navigator && ( <button onClick={handleImportContacts} style={{ width: '100%', padding: '10px', backgroundColor: '#eef2ff', color: '#4f46e5', border: '1px dashed #6366f1', borderRadius: '12px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '8px' }}><BookUser size={16} /> Importa dalla Rubrica del Telefono</button> )}
              <input placeholder="Nome (es. Mamma)" value={newContactName} onChange={e => setNewContactName(e.target.value)} style={{ padding: '10px', borderRadius: '8px', border: '1px solid #cbd5e1', outline: 'none' }} />
              <div style={{ display: 'flex', gap: '8px' }}>
                <input placeholder="Numero (+39...)" value={newContactPhone} onChange={e => setNewContactPhone(e.target.value)} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: '1px solid #cbd5e1', outline: 'none' }} />
                <select value={newContactApp} onChange={(e:any) => setNewContactApp(e.target.value)} style={{ padding: '10px', borderRadius: '8px', border: '1px solid #cbd5e1', outline: 'none' }}><option value="phone">Tel</option><option value="whatsapp">WA</option><option value="telegram">TG</option></select>
              </div>
              <button onClick={addContact} disabled={!newContactName || !newContactPhone} style={{ marginTop: '8px', padding: '12px', backgroundColor: '#9333ea', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 700, cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', opacity: (!newContactName || !newContactPhone) ? 0.5 : 1 }}><Plus size={18} /> Aggiungi</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;