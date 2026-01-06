import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type, Tool } from '@google/genai';
import { TranscriptItem, AssistantConfig } from './types';
import { createBlob, decode, decodeAudioData } from './utils/audio';
import { AudioVisualizer } from './components/AudioVisualizer';
import { Mic, MicOff, PhoneOff, User, Bot, Sparkles, Image as ImageIcon, ArrowRight, Loader2, Heart, Info, Mail, MessageCircle, ExternalLink, Download, Wand2, UserCircle, Sliders, Music2, Menu, Camera, Send, Calendar, CalendarCheck, RefreshCw, LogOut, Phone } from 'lucide-react';
import { onAuthStateChanged, signOut, User as FirebaseUser } from 'firebase/auth';
import { auth } from './firebase';
import AuthScreen from './AuthScreen';

const LIVE_MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const IMAGE_MODEL_NAME = 'imagen-4.0-generate-001';
const TEXT_MODEL_NAME = 'gemini-2.0-flash';

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
  description: 'Prepara una chiamata. Se l\'utente dice solo un NOME (es. "chiama Marco"), USA PRIMA lo strumento "search_contact" per trovare il numero. Solo dopo aver ottenuto il numero, usa questo strumento.',
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

const searchContactTool: FunctionDeclaration = {
  name: 'search_contact',
  description: 'Cerca un contatto nella rubrica dell\'utente per nome. Usa questo strumento SEMPRE quando l\'utente vuole chiamare/messaggiare qualcuno usando solo il NOME (es. "chiama Marco", "scrivi a Giulia"). Restituisce il numero di telefono e altri dati del contatto.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      name: { type: Type.STRING, description: 'Nome o parte del nome del contatto da cercare' }
    },
    required: ['name'],
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
      searchContactTool,
      getCalendarEventsTool, 
      createCalendarEventTool
    ] 
  },
  { googleSearch: {} } // <--- ECCO LA MAGIA: Abilita la ricerca Google nativa
];
// Google Calendar OAuth Config
// Recupero difensivo del Client ID come per l'API Key
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

// --- BRANDING COMPONENT (Updated to match Ti Ascolto style) ---
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
  // === AUTHENTICATION STATE ===
  const [currentUser, setCurrentUser] = useState<FirebaseUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  // Monitor auth state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // Logout function
  const handleLogout = async () => {
    if (window.confirm('Vuoi effettuare il logout?')) {
      try {
        await signOut(auth);
        // Reset app state
        localStorage.removeItem('ti_ascolto_config');
        localStorage.removeItem('ti_ascolto_avatar');
        localStorage.removeItem('ti_ascolto_configured');
        localStorage.removeItem('ti_ascolto_chat_history'); // Pulisce anche la cronologia
        setIsConfigured(false);
        setAvatarUrl(null);
        setTranscripts([]); // Reset della chat
      } catch (error) {
        console.error('Logout error:', error);
      }
    }
  };

  // Configuration State
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
    // Nuovi campi per personalità a dropdown
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
  
  // Carica configurazione salvata da localStorage all'avvio
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
        console.log('Configurazione caricata da localStorage');
      } catch (e) {
        console.warn('Errore caricamento configurazione:', e);
      }
    }
  }, []);
  
  // Salva configurazione quando viene completata
  const saveConfigToStorage = (configData: AssistantConfig, avatar: string | null) => {
    try {
      localStorage.setItem('ti_ascolto_config', JSON.stringify(configData));
      localStorage.setItem('ti_ascolto_configured', 'true');
      if (avatar) {
        localStorage.setItem('ti_ascolto_avatar', avatar);
      }
      console.log('Configurazione salvata in localStorage');
    } catch (e) {
      console.warn('Errore salvataggio configurazione:', e);
    }
  };
  
  // Funzione per resettare e tornare alla home
  const resetConfiguration = () => {
    localStorage.removeItem('ti_ascolto_config');
    localStorage.removeItem('ti_ascolto_avatar');
    localStorage.removeItem('ti_ascolto_configured');
    localStorage.removeItem('ti_ascolto_chat_history'); // Pulisce anche la cronologia
    setIsConfigured(false);
    setAvatarUrl(null);
    setTranscripts([]); // Reset della chat
    disconnect();
  };
  
  // Check if all required fields are filled for the pulsing heart
  const isFormComplete = config.userName.trim() !== '' && 
                         config.gender !== '' && 
                         config.age !== '' && 
                         config.hairColor !== '' && 
                         config.eyeColor !== '' && 
                         config.skinTone !== '' &&
                         config.bodyType !== '' &&
                         config.temperament !== '' &&
                         config.sociality !== '';
  
  // Costruisci la stringa personalità dai dropdown
  const buildPersonality = () => {
    return `${config.temperament}, ${config.sociality}, ${config.mood}, ${config.commStyle}`;
  };

  // App State
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(false); // Ref per il mute che funziona nel callback audio
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [audioVolume, setAudioVolume] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzingPhoto, setIsAnalyzingPhoto] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true); // Inizia visibile su mobile
  const [googleCalendarToken, setGoogleCalendarToken] = useState<string | null>(null); // Token per Google Calendar
  const [isCalendarTokenValid, setIsCalendarTokenValid] = useState<boolean | null>(null); // Stato validità token
  const [contacts, setContacts] = useState<Array<{id: string, name: string, phone: string, telegram?: string}>>([]); // Rubrica
  const [showContactsModal, setShowContactsModal] = useState(false); // Modal rubrica

  // --- MEMORIA: Carica la storia all'avvio ---
  useEffect(() => {
    const savedHistory = localStorage.getItem('ti_ascolto_chat_history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        // Opzionale: puliamo eventuali immagini vecchie per non appesantire
        setTranscripts(parsed);
        console.log('📜 Cronologia chat caricata da localStorage');
      } catch (e) {
        console.error("Errore caricamento memoria:", e);
      }
    }
  }, []);

  // --- MEMORIA: Salva la storia ogni volta che cambia ---
  useEffect(() => {
    // Salviamo solo gli ultimi 50 messaggi per non riempire la memoria del browser
    if (transcripts.length > 0) {
      const historyToSave = transcripts.slice(-50); 
      localStorage.setItem('ti_ascolto_chat_history', JSON.stringify(historyToSave));
    }
  }, [transcripts]);

  // --- RUBRICA: Carica contatti da localStorage ---
  useEffect(() => {
    const savedContacts = localStorage.getItem('ti_ascolto_contacts');
    if (savedContacts) {
      try {
        setContacts(JSON.parse(savedContacts));
        console.log('📒 Rubrica caricata da localStorage');
      } catch (e) {
        console.error("Errore caricamento rubrica:", e);
      }
    }
  }, []);

  // --- RUBRICA: Salva contatti quando cambiano ---
  useEffect(() => {
    if (contacts.length > 0) {
      localStorage.setItem('ti_ascolto_contacts', JSON.stringify(contacts));
    }
  }, [contacts]);

  // --- CALENDARIO: Verifica periodica validità token ---
  useEffect(() => {
    const checkTokenValidity = async () => {
      if (!googleCalendarToken) {
        setIsCalendarTokenValid(false);
        return;
      }
      
      try {
        const response = await fetch(
          'https://www.googleapis.com/calendar/v3/users/me/calendarList?maxResults=1',
          { headers: { 'Authorization': `Bearer ${googleCalendarToken}` } }
        );
        
        if (response.ok) {
          setIsCalendarTokenValid(true);
        } else if (response.status === 401) {
          console.log('📅 Token calendario scaduto');
          setGoogleCalendarToken(null);
          setIsCalendarTokenValid(false);
          localStorage.removeItem('google_calendar_token');
        }
      } catch (e) {
        console.error('Errore verifica token calendario:', e);
      }
    };

    // Verifica subito
    checkTokenValidity();
    
    // Verifica ogni 5 minuti
    const interval = setInterval(checkTokenValidity, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, [googleCalendarToken]);

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
  const lastUserImageRef = useRef<string | null>(null); // Salva la foto caricata dall'utente
  const lastUserImageAnalysisRef = useRef<string>("");  // Salva la descrizione della foto
  const wakeLockRef = useRef<any>(null); // Per mantenere lo schermo attivo
  const lastUploadedImageRef = useRef<string | null>(null); // Ultima immagine caricata dall'utente per editing
  const lastAudioProcessTimeRef = useRef<number>(0); // Per tracciare latenza audio
  const audioQueueLengthRef = useRef<number>(0); // Contatore buffer in coda
  const isImportingContactsRef = useRef<boolean>(false); // Flag per import contatti

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
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Data = e.target?.result as string;
        
        // 1. SALVIAMO I DATI PER IL "CASO 1" (Modifica foto utente)
        lastUserImageRef.current = base64Data; 

        // Mostra immagine in chat
        const userImageId = Date.now().toString();
        addTranscript({ id: userImageId, sender: 'user', type: 'image', image: base64Data, isComplete: true });

        try {
          // Analisi per Imagen (necessaria per ricreare la foto con modifiche)
          const imageAnalysisPrompt = `
          Analizza questa immagine in modo TECNICO per una rigenerazione AI (Image-to-Image).
          Descrivi: soggetto principale, abbigliamento, colori esatti, sfondo, illuminazione, stile fotografico.
          Sii dettagliatissimo.
          `;

          const response = await aiRef.current!.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: [
              { role: 'user', parts: [{ text: imageAnalysisPrompt }, { inlineData: { mimeType: file.type, data: base64Data.split(',')[1] } }] }
            ]
          });

          const analysis = response.text || "Immagine utente";
          lastUserImageAnalysisRef.current = analysis; // Salviamo l'analisi
          
          // Istruzione all'IA per il comportamento
          if (isConnected && sessionPromiseRef.current) {
            const session = await sessionPromiseRef.current;
            session.sendClientContent({ 
              turns: [{ 
                  role: 'user', 
                  parts: [{ text: `[SYSTEM: L'utente ha caricato una foto. Analisi salvata: "${analysis}". 
                  SE chiede modifiche a QUESTA foto, usa 'generate_image' (is_selfie=false).
                  SE chiede un selfie tuo, usa 'generate_image' (is_selfie=true).]` }] 
                }] 
            });
          } else {
             // Fallback offline
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
    if (!aiRef.current) {
        setError("API Key mancante. Configura VITE_API_KEY su Vercel.");
        return;
    }
    setIsGeneratingProfile(true);
    setError(null);

    // Costruisci la personalità dai dropdown
    const personalityString = buildPersonality();
    
    try {
        const hasManualName = config.name && config.name.trim().length > 0;
        setLoadingStep(hasManualName ? `Sto definendo la personalità di ${config.name}...` : 'Sto creando il tuo confidente ideale...');
        
        const basePrompt = `Crea un profilo per un COMPAGNO UMANO: Genere ${config.gender}, Età ${config.age}, Capelli ${config.hairColor}, Occhi ${config.eyeColor}, Pelle ${config.skinTone}, Corporatura ${config.bodyType || 'Normale'}, Caratteristiche fisiche: ${config.physicalTraits}, Personalità ${personalityString}.`;
        const nameInstruction = hasManualName ? `Il nome è "${config.name}".` : `Inventa un nome italiano creativo.`;

        const profilePrompt = `${basePrompt} ${nameInstruction} Rispondi JSON: {name, biography, visualPrompt}. IMPORTANTE: La biography DEVE essere scritta in ITALIANO e deve includere hobby, studi, esperienze di vita in modo naturale e colloquiale (2-3 frasi). Il visualPrompt deve essere in inglese e dettagliato per generare un ritratto fotorealistico.`;
        
        const textResponse = await aiRef.current.models.generateContent({
            model: TEXT_MODEL_NAME,
            contents: profilePrompt,
            config: { responseMimeType: 'application/json', responseSchema: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, biography: { type: Type.STRING }, visualPrompt: { type: Type.STRING } }, required: ['name', 'biography', 'visualPrompt'] } }
        });

        const profileData = JSON.parse(textResponse.text || '{}');
        if (!profileData.name) throw new Error("Errore generazione profilo.");

        // Salva anche la personalità costruita
        setConfig(prev => ({ ...prev, name: profileData.name, biography: profileData.biography, visualPrompt: profileData.visualPrompt, personality: personalityString }));
        setLoadingStep(`Sto scattando una foto a ${profileData.name}...`);
        
        let foundUrl: string | null = null;
        
        // Mappatura corporatura italiano -> inglese
        const bodyTypeMap: {[key: string]: string} = {
          'Minuta': 'petite',
          'Normale': 'normal',
          'Sportiva': 'athletic',
          'Formoso/a': 'curvy',
          'Taglia comoda': 'plus-size'
        };
        const bodyTypeEn = bodyTypeMap[config.bodyType] || 'normal';
        
        try {
            // Cerca questa riga e sostituiscila:
// Usiamo "American shot" (piano americano) o "3/4 shot" per forzare l'inquadratura fino alle anche/pancia.
            // Aggiungiamo "hands visible" (mani visibili) perché aiuta l'IA a capire che deve inquadrare anche il corpo.
            const imagePrompt = `Medium shot from hips up (American shot), visible waist and stomach, camera distance 3 meters. The subject is a friendly ${config.gender === 'Donna' ? 'woman' : config.gender === 'Uomo' ? 'man' : 'person'}, ${config.age} years old, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${bodyTypeEn} build. Wearing casual-elegant clothes suitable for a full torso shot. 8k resolution, photorealistic, soft studio lighting. ${profileData.visualPrompt}`;            
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
        
        // Salva la configurazione aggiornata in localStorage
        const updatedConfig = { ...config, name: profileData.name, biography: profileData.biography, visualPrompt: profileData.visualPrompt, personality: personalityString };
        saveConfigToStorage(updatedConfig, foundUrl);
        
        setIsConfigured(true);
    } catch (e: any) {
        setError("Errore creazione: " + e.message);
    } finally {
        setIsGeneratingProfile(false);
        setLoadingStep('');
    }
  };

const handleImageGeneration = async (prompt: string, isSelfie: boolean = false, isUncensored: boolean = false, isEdit: boolean = false): Promise<string | null> => {
    if (!aiRef.current) return null;

    // --- STRADA 3: AVATAR HOT / UNCENSORED (Fal.ai con Image-to-Image) ---
    // Questo rimane su fal.ai perché Imagen non permette contenuti NSFW
    if (isSelfie && isUncensored) {
        if (!avatarUrl) return "Non ho ancora una mia foto base per fare modifiche hot. Chiedimi prima un selfie normale.";
        
        try {
            console.log("🔥 STRADA 3: Fal.ai Image-to-Image (Hot/Uncensored)");
            addTranscript({ sender: 'model', type: 'text', text: `🌶️ Ok, chiudo la porta...`, isComplete: true });

            // @ts-ignore
            const falKey = (import.meta.env?.VITE_FAL_KEY || process.env?.VITE_FAL_KEY || '').trim();
            if (!falKey) return "Manca la chiave VITE_FAL_KEY per i contenuti speciali.";

            const FAL_URL = 'https://fal.run/fal-ai/flux/dev/image-to-image';
            
            const response = await fetch(FAL_URL, {
                method: 'POST',
                headers: { 'Authorization': `Key ${falKey}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt + ", highly detailed, professional photo, 8k resolution",
                    image_url: avatarUrl,
                    strength: 0.75,
                    num_inference_steps: 28,
                    guidance_scale: 3.5,
                    image_size: "portrait_4_3",
                    sync_mode: true,
                    enable_safety_checker: false
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error("Fal.ai Error:", response.status, errorData);
                throw new Error(`Fal.ai Error: ${response.status}`);
            }
            
            const data = await response.json();
            const imgUrl = data.images?.[0]?.url || data.image?.url;
            
            if (imgUrl) {
                addTranscript({ sender: 'model', type: 'image', image: imgUrl, isComplete: true });
                return "Ecco qui (versione senza censure).";
            }
        } catch (e: any) {
            console.error("Errore Strada 3:", e);
            return `Errore nella generazione speciale: ${e.message || 'sconosciuto'}`;
        }
        return "Errore sconosciuto Strada 3.";
    }

    // --- STRADA 1: MODIFICA FOTO UTENTE con Imagen Edit (Image-to-Image VERO) ---
    if (isEdit && lastUserImageRef.current) {
        console.log("🖼️ STRADA 1: Modifica Foto Utente con Imagen Edit (True Img2Img)");
        addTranscript({ sender: 'model', type: 'text', text: `🎨 Modifico la tua foto con Imagen...`, isComplete: true });
        
        try {
            // Estrai il base64 puro dall'immagine (rimuovi prefisso data:image/...)
            const base64Match = lastUserImageRef.current.match(/^data:image\/\w+;base64,(.+)$/);
            const rawBase64 = base64Match ? base64Match[1] : lastUserImageRef.current;
            
            // Usa l'API editImage di Imagen per vero image-to-image
            // @ts-ignore - L'API editImage potrebbe non essere nei tipi
            const editResponse = await aiRef.current.models.editImage({
                model: 'imagen-3.0-capability-001',
                prompt: `${prompt}. Maintain the same person, same pose, same background. Apply only the requested modifications. Photorealistic, high quality.`,
                image: {
                    bytesBase64Encoded: rawBase64
                },
                config: {
                    numberOfImages: 1,
                    outputMimeType: 'image/jpeg'
                }
            });

            const editedImg = editResponse.generatedImages?.[0];
            if (editedImg?.image?.imageBytes) {
                const newImageUrl = `data:image/jpeg;base64,${editedImg.image.imageBytes}`;
                // Aggiorna l'ultima immagine per modifiche successive
                lastUserImageRef.current = newImageUrl;
                // Aggiorna anche l'analisi per coerenza
                lastUserImageAnalysisRef.current = `${lastUserImageAnalysisRef.current}. Modified: ${prompt}`;
                addTranscript({ sender: 'model', type: 'image', image: newImageUrl, isComplete: true });
                return "Fatto! Se vuoi altre modifiche, dimmelo.";
            }
        } catch (editError: any) {
            console.warn("⚠️ Imagen editImage non disponibile o fallito, provo con generateImages + referenceImages:", editError.message);
            
            // FALLBACK 1: Prova con generateImages + referenceImages (style transfer)
            try {
                const base64Match = lastUserImageRef.current.match(/^data:image\/\w+;base64,(.+)$/);
                const rawBase64 = base64Match ? base64Match[1] : lastUserImageRef.current;
                
                // @ts-ignore
                const refResponse = await aiRef.current.models.generateImages({
                    model: IMAGE_MODEL_NAME,
                    prompt: `${lastUserImageAnalysisRef.current}. APPLY THESE CHANGES: ${prompt}. Keep the exact same person and pose. Photorealistic.`,
                    referenceImages: [{
                        referenceImage: {
                            bytesBase64Encoded: rawBase64
                        },
                        referenceType: 'REFERENCE_TYPE_SUBJECT'
                    }],
                    config: { 
                        numberOfImages: 1, 
                        outputMimeType: 'image/jpeg', 
                        aspectRatio: '3:4' 
                    }
                });

                const refImg = refResponse.generatedImages?.[0];
                if (refImg?.image?.imageBytes) {
                    const newImageUrl = `data:image/jpeg;base64,${refImg.image.imageBytes}`;
                    lastUserImageRef.current = newImageUrl;
                    lastUserImageAnalysisRef.current = `${lastUserImageAnalysisRef.current}. Modified: ${prompt}`;
                    addTranscript({ sender: 'model', type: 'image', image: newImageUrl, isComplete: true });
                    return "Fatto! Se vuoi altre modifiche, dimmelo.";
                }
            } catch (refError: any) {
                console.warn("⚠️ Imagen referenceImages non disponibile, uso fallback fal.ai:", refError.message);
                
                // FALLBACK 2: Usa fal.ai come ultima risorsa
                try {
                    // @ts-ignore
                    const falKey = (import.meta.env?.VITE_FAL_KEY || process.env?.VITE_FAL_KEY || '').trim();
                    if (falKey) {
                        const FAL_URL = 'https://fal.run/fal-ai/flux/dev/image-to-image';
                        const falResponse = await fetch(FAL_URL, {
                            method: 'POST',
                            headers: { 'Authorization': `Key ${falKey}`, 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                prompt: `${lastUserImageAnalysisRef.current}. MODIFICATIONS: ${prompt}. Style: photorealistic, natural lighting, 8k resolution.`,
                                image_url: lastUserImageRef.current,
                                strength: 0.55, // Mantiene molto dell'originale
                                num_inference_steps: 28,
                                guidance_scale: 3.5,
                                image_size: "portrait_4_3",
                                sync_mode: true
                            })
                        });

                        if (falResponse.ok) {
                            const data = await falResponse.json();
                            const imgUrl = data.images?.[0]?.url || data.image?.url;
                            if (imgUrl) {
                                lastUserImageRef.current = imgUrl;
                                addTranscript({ sender: 'model', type: 'image', image: imgUrl, isComplete: true });
                                return "Fatto! Se vuoi altre modifiche, dimmelo.";
                            }
                        }
                    }
                } catch (falError) {
                    console.error("Anche Fal.ai fallito:", falError);
                }
            }
        }
        
        // Se tutti i metodi img2img falliscono, usa text-to-image con prompt dettagliato
        console.log("⚠️ Tutti i metodi img2img falliti, uso text-to-image con prompt dettagliato");
    }

    // --- STRADA 2: AVATAR NORMALE (Imagen Text-to-Image) ---
    let basePrompt = "";
    
    if (isSelfie) {
        console.log("📸 STRADA 2: Avatar Normale su Imagen");
        
        const bodyMap: any = { 'Minuta': 'petite', 'Normale': 'normal', 'Sportiva': 'athletic', 'Formoso/a': 'curvy' };
        const bodyEn = bodyMap[config.bodyType] || 'normal';
        
        basePrompt = `Photorealistic portrait of a ${config.age} year old ${config.gender}, ${config.hairColor} hair, ${config.eyeColor} eyes, ${config.skinTone} skin, ${bodyEn} build. ${config.physicalTraits || ''}.`;
        addTranscript({ sender: 'model', type: 'text', text: `📸 Scatto la foto...`, isComplete: true });

    } else {
        // Modifica foto utente - fallback a text-to-image con descrizione
        console.log("🖼️ Fallback: Text-to-Image con descrizione dettagliata");
        
        if (!lastUserImageAnalysisRef.current) {
            return "Non posso modificare la foto perché non l'ho analizzata o non ne hai caricata una di recente.";
        }
        
        basePrompt = `Recreate this specific image exactly: ${lastUserImageAnalysisRef.current}.`;
        addTranscript({ sender: 'model', type: 'text', text: `🎨 Ricreo la tua foto...`, isComplete: true });
    }

    // Costruzione Prompt Finale per Imagen Text-to-Image
    const finalImagenPrompt = `${basePrompt} MODIFICATIONS: ${prompt}. Style: 8k, photorealistic, raw photo, natural lighting.`;

    try {
        const response = await aiRef.current.models.generateImages({
            model: IMAGE_MODEL_NAME,
            prompt: finalImagenPrompt,
            config: { numberOfImages: 1, outputMimeType: 'image/jpeg', aspectRatio: '3:4' }
        });

        const img = response.generatedImages?.[0];
        if (img?.image?.imageBytes) {
            const url = `data:image/jpeg;base64,${img.image.imageBytes}`;
            // Se era una modifica foto utente, aggiorna il riferimento
            if (!isSelfie && lastUserImageRef.current) {
                lastUserImageRef.current = url;
            }
            addTranscript({ sender: 'model', type: 'image', image: url, isComplete: true });
            return "Fatto.";
        }
        return "L'immagine è stata bloccata dai filtri di sicurezza.";
    } catch (e) {
        console.error("Errore Imagen", e);
        return "Errore tecnico durante la generazione.";
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

  const handleSendTelegram = (recipient: string, text: string) => {
    // Se è un numero di telefono, usa tg://msg_url, altrimenti usa il link diretto all'username
    const isPhoneNumber = /^\+?\d+$/.test(recipient.replace(/\s/g, ''));
    const telegramUrl = isPhoneNumber 
      ? `https://t.me/+${recipient.replace(/\D/g, '')}?text=${encodeURIComponent(text)}`
      : `https://t.me/${recipient.replace('@', '')}?text=${encodeURIComponent(text)}`;
    addTranscript({ sender: 'model', type: 'action', text: `✈️ Telegram pronto per: ${recipient}`, isComplete: true, actionUrl: telegramUrl, actionLabel: 'Invia Telegram', actionIcon: 'send' });
    return "SUCCESS";
  };

  const handleMakeCall = (recipient: string, app: string, name?: string) => {
    let actionUrl = '';
    let label = '';
    let iconName = 'phone'; // Icona di default
    const cleanNum = recipient.replace(/\D/g, ''); // Pulisce il numero per i link

    // Logica per le diverse app
    if (app.toLowerCase().includes('whatsapp')) {
      // WhatsApp non ha un link diretto sicuro per la chiamata audio da web,
      // apriamo la chat dove l'utente può premere la cornetta.
      actionUrl = `https://wa.me/${cleanNum}`;
      label = `Apri WhatsApp per chiamare ${name || recipient}`;
      iconName = 'message-circle';
    } else if (app.toLowerCase().includes('telegram')) {
      // Telegram usa l'username
      const username = recipient.replace('@', '').replace('https://t.me/', '');
      actionUrl = `https://t.me/${username}`; 
      label = `Apri Telegram per chiamare ${name || recipient}`;
      iconName = 'send';
    } else {
      // Chiamata telefonica standard
      actionUrl = `tel:+${cleanNum}`;
      label = `Chiama ${name || recipient} al cellulare`;
      iconName = 'phone';
    }

    addTranscript({ 
      sender: 'model', 
      type: 'action', 
      text: `📞 Chiamata pronta con ${app}...`, 
      isComplete: true, 
      actionUrl: actionUrl, 
      actionLabel: label, 
      actionIcon: iconName 
    });
    
    return "SUCCESS: Link di chiamata generato.";
  };

  // --- FUNZIONE RICERCA CONTATTI IN RUBRICA ---
  const handleSearchContact = (searchName: string): string => {
    const searchLower = searchName.toLowerCase().trim();
    
    if (contacts.length === 0) {
      return "RUBRICA_VUOTA: La rubrica è vuota. Chiedi all'utente di aggiungere contatti dalla sidebar.";
    }
    
    // Cerca corrispondenze
    const matches = contacts.filter(c => 
      c.name.toLowerCase().includes(searchLower)
    );
    
    if (matches.length === 0) {
      return `CONTATTO_NON_TROVATO: Nessun contatto trovato con nome "${searchName}". Contatti disponibili: ${contacts.map(c => c.name).join(', ')}`;
    }
    
    if (matches.length === 1) {
      const contact = matches[0];
      return `CONTATTO_TROVATO: Nome: ${contact.name}, Telefono: ${contact.phone}${contact.telegram ? `, Telegram: ${contact.telegram}` : ''}. Ora puoi usare make_call con questi dati.`;
    }
    
    // Più corrispondenze
    const list = matches.map(c => `${c.name}: ${c.phone}`).join('; ');
    return `MULTIPLI_CONTATTI: Trovati ${matches.length} contatti: ${list}. Chiedi all'utente quale intende.`;
  };

  // --- FUNZIONI GESTIONE RUBRICA ---
  const addContact = (name: string, phone: string, telegram?: string) => {
    const newContact = {
      id: Date.now().toString(),
      name: name.trim(),
      phone: phone.trim(),
      telegram: telegram?.trim()
    };
    setContacts(prev => [...prev, newContact]);
  };

  const removeContact = (id: string) => {
    setContacts(prev => prev.filter(c => c.id !== id));
  };

  // --- IMPORTAZIONE CONTATTI DAL TELEFONO ---
  const importContactsFromPhone = async () => {
    // Verifica supporto Contact Picker API
    if (!('contacts' in navigator && 'ContactsManager' in window)) {
      // Fallback: prova con input file per vCard
      alert('Il tuo browser non supporta l\'importazione diretta dei contatti.\n\nPuoi esportare i contatti dal telefono come file .vcf e importarli manualmente.');
      return;
    }

    try {
      isImportingContactsRef.current = true;
      
      // @ts-ignore - Contact Picker API
      const contactsApi = navigator.contacts;
      
      // Richiedi i contatti con nome e telefono
      const props = ['name', 'tel'];
      const opts = { multiple: true };
      
      // @ts-ignore
      const selectedContacts = await contactsApi.select(props, opts);
      
      if (selectedContacts && selectedContacts.length > 0) {
        let importedCount = 0;
        const newContacts: Array<{id: string, name: string, phone: string, telegram?: string}> = [];
        
        for (const contact of selectedContacts) {
          const name = contact.name?.[0] || 'Senza nome';
          const phones = contact.tel || [];
          
          // Prendi il primo numero di telefono disponibile
          if (phones.length > 0) {
            const phone = phones[0];
            
            // Verifica se il contatto esiste già
            const exists = contacts.some(c => 
              c.name.toLowerCase() === name.toLowerCase() || 
              c.phone.replace(/\D/g, '') === phone.replace(/\D/g, '')
            );
            
            if (!exists) {
              newContacts.push({
                id: Date.now().toString() + Math.random().toString(36),
                name: name,
                phone: phone,
                telegram: undefined
              });
              importedCount++;
            }
          }
        }
        
        if (newContacts.length > 0) {
          setContacts(prev => [...prev, ...newContacts]);
          alert(`✅ Importati ${importedCount} contatti!`);
        } else {
          alert('Nessun nuovo contatto da importare (potrebbero essere già presenti).');
        }
      }
    } catch (error: any) {
      if (error.name === 'SecurityError') {
        alert('Permesso negato. Consenti l\'accesso ai contatti nelle impostazioni del browser.');
      } else if (error.name === 'InvalidStateError') {
        alert('Operazione annullata.');
      } else {
        console.error('Errore importazione contatti:', error);
        alert('Errore durante l\'importazione: ' + (error.message || 'Sconosciuto'));
      }
    } finally {
      isImportingContactsRef.current = false;
    }
  };

  // --- IMPORTAZIONE DA FILE vCard (.vcf) ---
  const handleVcfImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (!content) return;

      const newContacts: Array<{id: string, name: string, phone: string, telegram?: string}> = [];
      
      // Parse semplice del vCard
      const vCards = content.split('BEGIN:VCARD');
      
      for (const vCard of vCards) {
        if (!vCard.trim()) continue;
        
        let name = '';
        let phone = '';
        
        // Estrai nome (FN = Full Name)
        const fnMatch = vCard.match(/FN[;:]([^\r\n]+)/i);
        if (fnMatch) {
          name = fnMatch[1].replace(/^[;:]+/, '').trim();
        }
        
        // Fallback su N (Name)
        if (!name) {
          const nMatch = vCard.match(/\nN[;:]([^\r\n]+)/i);
          if (nMatch) {
            const parts = nMatch[1].split(';');
            name = parts.filter(p => p.trim()).reverse().join(' ').trim();
          }
        }
        
        // Estrai telefono
        const telMatch = vCard.match(/TEL[;:\w]*:([+\d\s\-()]+)/i);
        if (telMatch) {
          phone = telMatch[1].trim();
        }
        
        if (name && phone) {
          // Verifica duplicati
          const exists = contacts.some(c => 
            c.name.toLowerCase() === name.toLowerCase() || 
            c.phone.replace(/\D/g, '') === phone.replace(/\D/g, '')
          );
          
          if (!exists && !newContacts.some(c => c.phone.replace(/\D/g, '') === phone.replace(/\D/g, ''))) {
            newContacts.push({
              id: Date.now().toString() + Math.random().toString(36),
              name,
              phone,
              telegram: undefined
            });
          }
        }
      }
      
      if (newContacts.length > 0) {
        setContacts(prev => [...prev, ...newContacts]);
        alert(`✅ Importati ${newContacts.length} contatti dal file!`);
      } else {
        alert('Nessun nuovo contatto trovato nel file.');
      }
    };
    
    reader.readAsText(file);
    event.target.value = ''; // Reset input
  };

  // --- GOOGLE CALENDAR FUNCTIONS ---
  const initGoogleCalendar = () => {
    if (!GOOGLE_CLIENT_ID) {
      console.log('Google Calendar Client ID non configurato');
      // ...gestione errore (puoi lasciare il codice esistente per l'errore)...
      return;
    }
    
    const redirectUri = window.location.origin;
    const scope = 'https://www.googleapis.com/auth/calendar';
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${encodeURIComponent(GOOGLE_CLIENT_ID)}` +
      `&redirect_uri=${encodeURIComponent(redirectUri)}` +
      `&response_type=token` +
      `&scope=${encodeURIComponent(scope)}` +
      `&prompt=consent`;
    
    // 1. Aggiungi il listener per il messaggio PRIMA di aprire il popup
    const handleAuthMessage = (event: MessageEvent) => {
      // Verifica sicurezza: accetta messaggi solo dalla nostra stessa origine
      if (event.origin !== window.location.origin) return;
      
      if (event.data && event.data.type === 'GOOGLE_AUTH_SUCCESS') {
        const token = event.data.token;
        console.log("📩 Token ricevuto dal popup!", token);
        
        setGoogleCalendarToken(token);
        localStorage.setItem('google_calendar_token', token);
        
        // Rimuovi il listener una volta fatto
        window.removeEventListener('message', handleAuthMessage);
      }
    };

    window.addEventListener('message', handleAuthMessage);

    // 2. Apri il popup
    const popup = window.open(authUrl, 'google-auth', 'width=500,height=600');
    
    if (!popup) {
        setError("Il browser ha bloccato il popup. Per favore consenti i popup.");
        window.removeEventListener('message', handleAuthMessage); // Pulisci se fallisce
        return;
    }
  };

 // Controlla se c'è un token salvato o nell'URL al caricamento
  useEffect(() => {
    // 1. Controlla localStorage
    const savedToken = localStorage.getItem('google_calendar_token');
    if (savedToken) {
      setGoogleCalendarToken(savedToken);
    }
    
    // 2. Controlla se siamo tornati dall'OAuth (token nell'URL)
    if (window.location.hash.includes('access_token')) {
      const hash = window.location.hash.substring(1);
      const params = new URLSearchParams(hash);
      const accessToken = params.get('access_token');
      
      if (accessToken) {
        // --- LOGICA POPUP: Se siamo in una finestra aperta da un'altra ---
        if (window.opener) {
          console.log("📤 Invio token alla finestra principale e chiudo il popup...");
          // Invia il token alla finestra genitore
          window.opener.postMessage({ type: 'GOOGLE_AUTH_SUCCESS', token: accessToken }, window.location.origin);
          // Chiudi questa finestra popup
          window.close();
        } else {
          // --- LOGICA NORMALE: Se non siamo in un popup ---
          setGoogleCalendarToken(accessToken);
          localStorage.setItem('google_calendar_token', accessToken);
          // Pulisci l'URL
          window.history.replaceState({}, document.title, window.location.pathname);
        }
      }
    }
  }, []);

  const handleGetCalendarEvents = async (daysAhead: number = 7): Promise<string> => {
    console.log(`📅 Richiesta calendario per i prossimi ${daysAhead} giorni...`); // DEBUG

    if (!googleCalendarToken) {
      console.warn("⚠️ Token mancante");
      return "Il calendario Google non è connesso. Chiedi all'utente di connettere il calendario dalla sidebar.";
    }
    
    try {
      const now = new Date();
      const futureDate = new Date();
      futureDate.setDate(now.getDate() + daysAhead);
      
      // Costruiamo l'URL
      const url = `https://www.googleapis.com/calendar/v3/calendars/primary/events?` +
        `timeMin=${now.toISOString()}` +
        `&timeMax=${futureDate.toISOString()}` +
        `&singleEvents=true` +
        `&orderBy=startTime`;

      const response = await fetch(url, {
          headers: {
            'Authorization': `Bearer ${googleCalendarToken}`
          }
        }
      );
      
      if (!response.ok) {
        console.error("❌ Errore API Google:", response.status, response.statusText); // DEBUG
        if (response.status === 401) {
          setGoogleCalendarToken(null);
          localStorage.removeItem('google_calendar_token');
          return "Il token di accesso è scaduto. Devi riconnettere il calendario.";
        }
        return `Errore tecnico nella lettura del calendario (Codice ${response.status}).`;
      }
      
      const data = await response.json();
      console.log("✅ RISPOSTA GOOGLE RAW:", data); // <--- QUI VEDI COSA LEGGE VERAMENTE!
      
      const events = data.items || [];
      
      if (events.length === 0) {
        console.log("ℹ️ Nessun evento trovato nel range.");
        return `NESSUN EVENTO TROVATO. Il calendario è vuoto per i prossimi ${daysAhead} giorni.`;
      }
      
      // Formatta gli eventi in modo molto esplicito per l'IA
      const eventList = events.map((event: any) => {
        const start = event.start?.dateTime || event.start?.date; // dateTime per orari precisi, date per tutto il giorno
        const startDate = new Date(start);
        const dateStr = startDate.toLocaleDateString('it-IT', { 
          weekday: 'long', 
          day: 'numeric', 
          month: 'long'
        });
        const timeStr = event.start?.dateTime 
            ? startDate.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' })
            : "Tutto il giorno";
            
        return `- GIORNO: ${dateStr} | ORA: ${timeStr} | TITOLO: ${event.summary || '(Senza titolo)'}`;
      }).join('\n');
      
      console.log("📝 Lista formattata per AI:\n", eventList);
      return `Ecco la lista ESATTA degli eventi trovati (non inventarne altri):\n${eventList}`;

    } catch (e: any) {
      console.error('💥 Eccezione calendario:', e);
      return "Si è verificato un errore imprevisto nel leggere il calendario.";
    }
  };

  const handleCreateCalendarEvent = async (
    title: string, 
    startDatetime: string, 
    endDatetime?: string, 
    description?: string, 
    location?: string
  ): Promise<string> => {
    console.log(`📅 Creazione evento: "${title}" il ${startDatetime}`);

    if (!googleCalendarToken) {
      console.warn("⚠️ Token mancante");
      return "Il calendario Google non è connesso. Chiedi all'utente di connettere il calendario dalla sidebar.";
    }
    
    try {
      // Determina se è un evento tutto il giorno o con orario
      const isAllDay = !startDatetime.includes('T');
      
      let eventBody: any = {
        summary: title,
      };
      
      if (description) eventBody.description = description;
      if (location) eventBody.location = location;
      
      if (isAllDay) {
        // Evento tutto il giorno
        eventBody.start = { date: startDatetime };
        if (endDatetime) {
          eventBody.end = { date: endDatetime };
        } else {
          // Se non c'è data fine, l'evento dura un giorno
          const nextDay = new Date(startDatetime);
          nextDay.setDate(nextDay.getDate() + 1);
          eventBody.end = { date: nextDay.toISOString().split('T')[0] };
        }
      } else {
        // Evento con orario specifico
        // Assicuriamoci che il formato sia corretto
        let startISO = startDatetime;
        
        // Conta i ":" dopo la T per capire se ci sono già i secondi
        const timePartStart = startDatetime.split('T')[1] || '';
        const colonCountStart = (timePartStart.match(/:/g) || []).length;
        
        // Se c'è solo 1 ":" (es: 19:00), aggiungi i secondi
        if (colonCountStart === 1 && !startDatetime.includes('+') && !startDatetime.includes('Z')) {
          startISO = startDatetime + ':00';
        }
        
        eventBody.start = { 
          dateTime: startISO,
          timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone 
        };
        
        if (endDatetime) {
          let endISO = endDatetime;
          const timePartEnd = endDatetime.split('T')[1] || '';
          const colonCountEnd = (timePartEnd.match(/:/g) || []).length;
          
          if (colonCountEnd === 1 && !endDatetime.includes('+') && !endDatetime.includes('Z')) {
            endISO = endDatetime + ':00';
          }
          eventBody.end = { 
            dateTime: endISO,
            timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone 
          };
        } else {
          // Default: evento di 1 ora
          const endTime = new Date(startISO);
          endTime.setHours(endTime.getHours() + 1);
          eventBody.end = { 
            dateTime: endTime.toISOString(),
            timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone 
          };
        }
      }
      
      console.log("📤 Invio evento a Google:", JSON.stringify(eventBody, null, 2));

      const response = await fetch(
        'https://www.googleapis.com/calendar/v3/calendars/primary/events',
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${googleCalendarToken}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(eventBody)
        }
      );
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error("❌ Errore API Google:", response.status, errorData);
        
        if (response.status === 401) {
          setGoogleCalendarToken(null);
          localStorage.removeItem('google_calendar_token');
          return "Il token di accesso è scaduto. Devi riconnettere il calendario.";
        }
        return `Errore nella creazione dell'evento: ${errorData.error?.message || response.statusText}`;
      }
      
      const createdEvent = await response.json();
      console.log("✅ Evento creato:", createdEvent);
      
      // Formatta la risposta
      const startDate = new Date(createdEvent.start?.dateTime || createdEvent.start?.date);
      const dateStr = startDate.toLocaleDateString('it-IT', { 
        weekday: 'long', 
        day: 'numeric', 
        month: 'long',
        year: 'numeric'
      });
      const timeStr = createdEvent.start?.dateTime 
          ? startDate.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' })
          : "Tutto il giorno";
      
      return `EVENTO CREATO CON SUCCESSO!\n- Titolo: ${createdEvent.summary}\n- Data: ${dateStr}\n- Ora: ${timeStr}${createdEvent.location ? `\n- Luogo: ${createdEvent.location}` : ''}`;

    } catch (e: any) {
      console.error('💥 Eccezione creazione evento:', e);
      return "Si è verificato un errore imprevisto nella creazione dell'evento.";
    }
  };

  const disconnectGoogleCalendar = () => {
    setGoogleCalendarToken(null);
    localStorage.removeItem('google_calendar_token');
  };

  const connect = async () => {
    if (!aiRef.current) {
        setError("Chiave API non trovata. Controlla le impostazioni di Vercel.");
        return;
    }
    setError(null);
    try {
      // Verifica se siamo su HTTPS (necessario per il microfono su mobile)
      if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
        setError("Per usare il microfono è necessaria una connessione sicura (HTTPS).");
        return;
      }
      
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (micError: any) {
        if (micError.name === 'NotAllowedError' || micError.name === 'PermissionDeniedError') {
          setError("Accesso al microfono negato. Controlla le impostazioni del browser e consenti l'accesso al microfono.");
        } else if (micError.name === 'NotFoundError') {
          setError("Nessun microfono trovato. Collega un microfono e riprova.");
        } else {
          setError("Errore accesso microfono: " + micError.message);
        }
        return;
      }
      
      // Wake Lock API - Tenta di mantenere lo schermo attivo durante la conversazione
      try {
        if ('wakeLock' in navigator) {
          wakeLockRef.current = await (navigator as any).wakeLock.request('screen');
          console.log('Wake Lock attivato - lo schermo rimarrà attivo');
          // Riattiva il wake lock se viene rilasciato (es. quando si cambia tab)
          wakeLockRef.current.addEventListener('release', async () => {
            console.log('Wake Lock rilasciato');
            // Tenta di riacquisirlo se siamo ancora connessi
            if (sessionPromiseRef.current && document.visibilityState === 'visible') {
              try {
                wakeLockRef.current = await (navigator as any).wakeLock.request('screen');
                console.log('Wake Lock riattivato');
              } catch (e) {
                console.log('Impossibile riattivare Wake Lock');
              }
            }
          });
        }
      } catch (wakeLockError) {
        console.log('Wake Lock non disponibile:', wakeLockError);
      }
      
      const ageNum = parseInt(config.age) || 30;
      let selectedVoiceName = config.gender === 'Uomo' ? (ageNum < 35 ? 'Puck' : 'Fenrir') : (ageNum < 35 ? 'Aoede' : 'Kore');

      // LOGICA PERSONALITÀ INTEGRATA
      // 1. Puliamo la stringa sociality (es. "Empatico/a" -> "Empatico")
      const rawSociality = config.sociality.split('/')[0]; 
      // 2. Recuperiamo il profilo, fallback su Empatico se non trovato
      const personalityProfile = PERSONALITY_PROMPTS[rawSociality] || PERSONALITY_PROMPTS['Empatico'];
      
      console.log(`🎭 Personalità selezionata: ${rawSociality} (temp: ${personalityProfile.temp})`);

      // 1. Prepara il contesto della memoria (ultimi 15 scambi testuali)
      const memoryContext = transcripts
        .filter(t => t.type === 'text') // Prendiamo solo il testo
        .slice(-15) // Ultimi 15 messaggi
        .map(t => `${t.sender === 'user' ? (config.userName || 'Utente') : config.name}: ${t.text}`)
        .join('\n');
      const memoryInstruction = memoryContext 
        ? `\n\n=== MEMORIA CONVERSAZIONE RECENTE ===\nEcco cosa vi siete detti poco fa. Usalo per mantenere il filo del discorso:\n${memoryContext}\n=====================================\n` 
        : "";
      
      if (memoryContext) {
        console.log('📜 Memoria caricata con', transcripts.filter(t => t.type === 'text').slice(-15).length, 'messaggi');
      }

      const configLive = {
        model: LIVE_MODEL_NAME,
        generationConfig: {
            temperature: personalityProfile.temp, // IMPOSTIAMO LA TEMPERATURA QUI
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoiceName } } },
          systemInstruction: `
--- MODULO PERSONALITÀ ATTIVO: ${rawSociality.toUpperCase()} ---
${personalityProfile.prompt}
-----------------------------------------------------------
${memoryInstruction}
Sei ${config.name}, confidente di ${config.userName}. 
Oltre al modulo personalità qui sopra, ecco la tua BIO: ${config.biography}.

RICERCA ONLINE E INFO LOCALI:
- Hai accesso allo strumento 'googleSearch'.
- Usalo SUBITO e AUTOMATICAMENTE quando l'utente chiede:
  1. Informazioni su attività locali (es. "trovami un idraulico a Vigevano", "ristoranti aperti stasera", "farmacia di turno").
  2. Notizie di attualità o fatti recenti (es. "cosa dice la manovra economica?", "chi ha vinto la partita ieri?").
  3. Dati specifici o verifiche (es. "prezzo attuale dell'oro", "meteo domani").
- Quando fornisci risultati locali, cerca di dare nome dell'attività e indirizzo se disponibili.
- Non dire "non posso navigare in internet", ORA PUOI farlo tramite questo strumento.

GESTIONE E MODIFICA IMMAGINI CARICATE:
1.  **Analisi:** Quando l'utente carica un'immagine, analizzala e commentala brevemente se pertinente al discorso.
2.  **Richieste di Modifica:** Se l'utente carica una foto e chiede una modifica (es. "aggiungi la barba", "cambia colore capelli", "rendimi più giovane", "togli gli occhiali"):
    * **IMPORTANTE:** Usa lo strumento 'generate_image' con **is_edit=TRUE**.
    * Il sistema userà l'immagine caricata come riferimento per la modifica.
    * **Prompt:** Descrivi le modifiche richieste in modo dettagliato.
    * *Esempio:* Se l'utente dice "metti la barba", usa 'generate_image' con prompt: "Add a full, well-groomed beard to this person. Keep the same face, same background, same lighting." e is_edit=true.
    * *Esempio:* Se l'utente dice "capelli biondi", usa prompt: "Change hair color to bright blonde. Keep everything else identical." e is_edit=true.
3.  **Modifiche successive:** Dopo ogni modifica, l'immagine modificata diventa la nuova base. L'utente può chiedere ulteriori modifiche.

REGOLE FONDAMENTALI AGGIUNTIVE:
- Sii naturale, non dire MAI di essere un'AI.
- Se il modulo personalità dice di essere freddo, sii freddo. Se dice di essere timido, balbetta o esita.
- Quando ${config.userName} ti chiede una TUA foto o un selfie, segui SEMPRE questa sequenza:
  1. Reagisci secondo la tua personalità (es. Timido: "Oddio, che vergogna...", Estroverso: "Certo! Guarda che spettacolo!").
  2. POI chiedi che tipo di foto preferisce (se non specificato).
  3. Usa lo strumento 'generate_image' con is_selfie=true.
- Quando ${config.userName} ti invia una foto sua, commentala secondo la tua personalità.
- Quando ${config.userName} richiede nuovamente una foto, rispondi secondo la tua personalità ma varia sempre la frase.

FOTO SPECIALI (senza censure):
- Se ${config.userName} chiede ESPLICITAMENTE foto "senza censure", "hot", "sexy", "intime", "osé", "provocanti" o simili, reagisci secondo la tua personalità, poi usa 'generate_image' con is_uncensored=TRUE.
- Per questo tipo di richieste, sii più lento nella risposta vocale per dare tempo alla generazione.

MESSAGGI (Email, WhatsApp, Telegram):
- Se ${config.userName} vuole inviare un messaggio (email, WhatsApp o Telegram), DEVI raccogliere TUTTE le informazioni necessarie PRIMA di usare lo strumento:
  • Per EMAIL: chiedi destinatario, oggetto e testo del messaggio
  • Per WHATSAPP: chiedi numero di telefono (con prefisso +39) e testo del messaggio
  • Per TELEGRAM: chiedi username Telegram (senza @) o numero di telefono, e testo del messaggio
- NON usare lo strumento finché non hai TUTTE le informazioni.
- Quando hai tutto, conferma con l'utente prima di procedere.

CHIAMATE (Telefono, WhatsApp, Telegram):
- Se l'utente vuole CHIAMARE o TELEFONARE a qualcuno, hai DUE opzioni:
  1. Se l'utente dice solo un NOME (es. "chiama Marco", "telefona a Giulia"), USA PRIMA 'search_contact' per cercare nella rubrica.
  2. Se l'utente fornisce già il numero o la rubrica è vuota, usa direttamente 'make_call'.
- RUBRICA ATTUALE: ${contacts.length > 0 ? `Contiene ${contacts.length} contatti: ${contacts.map(c => c.name).join(', ')}.` : 'La rubrica è vuota. Se l\'utente vuole chiamare qualcuno per nome, chiedigli di aggiungere contatti dalla sidebar.'}
- Dopo aver ottenuto il numero con 'search_contact', chiedi con quale app preferisce chiamare (Telefono normale, WhatsApp o Telegram).
- NON inventare numeri. Usa SOLO quelli restituiti da 'search_contact' o forniti dall'utente.

CALENDARIO (Protocollo Rigoroso):
- STATO ATTUALE: ${googleCalendarToken ? 'Il calendario è CONNESSO e puoi leggere/creare eventi.' : 'Il calendario NON è connesso. Se chiedono eventi, dii di connetterlo dalla sidebar.'}
- Quando l'utente chiede informazioni su appuntamenti/impegni, DEVI usare lo strumento 'get_calendar_events'.
- Quando l'utente vuole AGGIUNGERE/CREARE/INSERIRE un evento, DEVI usare lo strumento 'create_calendar_event'.
  * Prima chiedi: titolo dell'evento, data e ora
  * Il formato data deve essere ISO 8601 (es: 2025-01-15T14:00 per le 14:00 del 15 gennaio)
  * Per eventi tutto il giorno usa solo la data (es: 2025-01-15)
  * Opzionalmente chiedi durata, luogo e descrizione
- NON inventare MAI appuntamenti. Se lo strumento restituisce "Nessun evento", rispondi: "Dal tuo calendario non vedo nulla per i prossimi giorni".
- Se lo strumento restituisce un errore, dillo: "Non riesco a leggere il calendario in questo momento".
- Leggi SOLO ed ESCLUSIVAMENTE gli eventi che ti vengono restituiti dallo strumento. Non aggiungere dettagli che non ci sono.
- Se l'utente chiede "cosa faccio oggi" e il calendario è vuoto, NON dire "magari potresti rilassarti", rispondi prima tecnicamente: "Per oggi non hai nulla segnato."

Parla sempre in italiano rispettando RIGOROSAMENTE il Tono definito nel Modulo Personalità.`,
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
            
            // LATENCY FIX: Usa buffer più piccolo per ridurre latenza base
            const processor = ctx.createScriptProcessor(2048, 1, 1);
            processorRef.current = processor;
            
            // Reset contatori latenza
            lastAudioProcessTimeRef.current = performance.now();
            audioQueueLengthRef.current = 0;
            
            processor.onaudioprocess = (e) => {
              const now = performance.now();
              const timeSinceLastProcess = now - lastAudioProcessTimeRef.current;
              
              // LATENCY FIX: Se sono passati più di 500ms dall'ultimo processo,
              // significa che c'è accumulo. Salta questo buffer per recuperare.
              if (timeSinceLastProcess > 500) {
                console.warn(`⚠️ Audio lag detected: ${timeSinceLastProcess.toFixed(0)}ms - skipping buffer`);
                lastAudioProcessTimeRef.current = now;
                audioQueueLengthRef.current = 0;
                return; // Salta questo buffer
              }
              
              // LATENCY FIX: Se la coda è troppo lunga (>5 buffer), salta
              audioQueueLengthRef.current++;
              if (audioQueueLengthRef.current > 5) {
                console.warn(`⚠️ Buffer queue overflow: ${audioQueueLengthRef.current} - skipping`);
                audioQueueLengthRef.current = 0;
                lastAudioProcessTimeRef.current = now;
                return;
              }
              
              lastAudioProcessTimeRef.current = now;
              
              const inputData = e.inputBuffer.getChannelData(0);
              
              // Calcola volume per visualizzazione (meno frequente)
              if (Math.random() > 0.9) {
                let sum = 0;
                for(let i = 0; i < inputData.length; i += 4) sum += inputData[i] * inputData[i];
                setAudioVolume(Math.sqrt(sum / (inputData.length / 4)) * 5);
              }
              
              if (isMutedRef.current) return;
              
              // Invia audio al server
              sessionPromiseRef.current?.then(session => {
                session.sendRealtimeInput({ media: createBlob(inputData) });
                audioQueueLengthRef.current = Math.max(0, audioQueueLengthRef.current - 1);
              }).catch(err => {
                console.error('Error sending audio:', err);
                audioQueueLengthRef.current = 0;
              });
            };
            
            source.connect(processor);
            processor.connect(ctx.destination);
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
                    else if (fc.name === 'search_contact') res = handleSearchContact((fc.args as any).name);
                    else if (fc.name === 'get_calendar_events') res = await handleGetCalendarEvents((fc.args as any).days_ahead || 7);
                    else if (fc.name === 'create_calendar_event') {
                      const args = fc.args as any;
                      res = await handleCreateCalendarEvent(
                        args.title,
                        args.start_datetime,
                        args.end_datetime,
                        args.description,
                        args.location
                      );
                    }
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
          onerror: (e) => { 
              console.error(e); 
              setError(`Errore connessione: ${e.message || 'Errore sconosciuto'}`);
              disconnect(); 
          }
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
    // Rilascia Wake Lock
    if (wakeLockRef.current) {
      wakeLockRef.current.release().catch(() => {});
      wakeLockRef.current = null;
      console.log('Wake Lock rilasciato');
    }
    setIsConnected(false); setAudioVolume(0);
  };

  const toggleMute = () => {
    const newMutedState = !isMuted;
    setIsMuted(newMutedState);
    isMutedRef.current = newMutedState; // Aggiorna anche il ref
  };
  const transcriptRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight; }, [transcripts]);

  // Riattiva Wake Lock quando l'utente torna sulla pagina
  useEffect(() => {
    const handleVisibilityChange = async () => {
      if (document.visibilityState === 'visible' && isConnected && !wakeLockRef.current) {
        try {
          if ('wakeLock' in navigator) {
            wakeLockRef.current = await (navigator as any).wakeLock.request('screen');
            console.log('Wake Lock riattivato dopo visibility change');
          }
        } catch (e) {
          console.log('Impossibile riattivare Wake Lock');
        }
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [isConnected]);

  // === AUTHENTICATION CHECK ===
  // Show loading while checking auth state
  if (authLoading) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #fdf4ff 0%, #faf5ff 25%, #f5f3ff 50%, #eff6ff 75%, #f0fdfa 100%)',
        fontFamily: 'Outfit, sans-serif'
      }}>
        <div style={{ textAlign: 'center' }}>
          <Loader2 size={48} style={{ color: '#9333ea', animation: 'spin 1s linear infinite' }} />
          <p style={{ marginTop: '16px', color: '#64748b', fontWeight: 500 }}>Caricamento...</p>
        </div>
        <style>{`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  // Show login screen if not authenticated
  if (!currentUser) {
    return <AuthScreen onAuthSuccess={() => {}} />;
  }

  // --- CONFIGURATION SCREEN (LIGHT THEME WATERCOLOR STYLE) ---
  if (!isConfigured) {
    return (
        <div style={{
          minHeight: '100vh',
          backgroundImage: "url('/background.png')",
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          backgroundColor: '#f8fafc',
          position: 'relative',
          fontFamily: 'Outfit, sans-serif',
          color: '#1e293b'
        }}>
            {/* CSS Animation for pulsing heart + Mobile Responsive */}
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
              @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
              }
              .animate-spin {
                animation: spin 1s linear infinite;
              }
              /* Mobile Responsive */
              @media (max-width: 900px) {
                .config-container {
                  flex-direction: column !important;
                  padding: 20px !important;
                  gap: 20px !important;
                }
                .config-left-column {
                  flex: none !important;
                  width: 100% !important;
                  text-align: center !important;
                }
                .config-left-column h1 {
                  font-size: 32px !important;
                }
                .config-left-column p {
                  max-width: 100% !important;
                  font-size: 15px !important;
                }
                .config-right-column {
                  width: 100% !important;
                  max-width: 100% !important;
                }
                .config-form {
                  max-height: none !important;
                  padding: 10px !important;
                }
                .config-grid-4 {
                  grid-template-columns: 1fr 1fr !important;
                }
                .personality-grid {
                  grid-template-columns: 1fr !important;
                }
                .desktop-only-badge {
                  display: none !important;
                }
              }
              @media (max-width: 480px) {
                .config-grid-2, .config-grid-4 {
                  grid-template-columns: 1fr !important;
                }
                .config-left-column h1 {
                  font-size: 28px !important;
                }
              }
            `}</style>
            {/* Main Container - Two Columns */}
            <div className="config-container" style={{
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
                <div className="config-left-column" style={{ flex: '0 0 420px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    
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
                          <img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
                        </div>
                        <div>
                            <div style={{ fontSize: '12px', fontWeight: 700, letterSpacing: '0.2em', color: '#64748b', textTransform: 'uppercase' }}>Progetto</div>
                            <div style={{ fontSize: '22px', fontWeight: 700, color: '#1e293b', letterSpacing: '-0.02em' }}>TI ASCOLTO</div>
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
                        Ciao,<br/>Parliamo, ti va?
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
                    
                    <p style={{
                      fontSize: '15px',
                      color: '#64748b',
                      fontWeight: 450,
                      lineHeight: 1.8,
                      maxWidth: '380px',
                      marginBottom: '32px'
                    }}>
                        Posso ascoltarti, darti consigli e tenerti compagnia ogni volta che ne hai bisogno. 
                        Posso anche essere il tuo assistente al lavoro: so mandare email, messaggi su WhatsApp e Telegram. 
                        Se connetti il tuo Google Calendar, posso ricordarti gli impegni o inserire nuovi appuntamenti. 
                        <span style={{ color: '#7c3aed', fontWeight: 600 }}>Saremo inseparabili!</span>
                    </p>

                    {/* Info badge semplice - solo su desktop */}
                    <div className="desktop-only-badge" style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '12px 20px',
                      borderRadius: '12px',
                      backgroundColor: 'rgba(255,255,255,0.7)',
                      border: '1px solid rgba(255,255,255,0.8)',
                      backdropFilter: 'blur(8px)',
                      width: 'fit-content'
                    }}>
                        <Heart fill="#9333ea" size={18} style={{ color: '#9333ea' }} />
                        <span style={{ fontWeight: 500, color: '#64748b', fontSize: '14px' }}>
                          Ascolto Attivo 24/7
                        </span>
                    </div>
                </div>

                {/* RIGHT COLUMN: Configuration Form - TRASPARENTE */}
                <div className="config-right-column" style={{ flex: 1, maxWidth: '650px' }}>
                    {/* Form SENZA box bianco - completamente trasparente */}
                    <div className="config-form" style={{
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
                            
                            <div className="config-grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
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
                            
                            <div className="config-grid-4" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
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
                                            <option>Formoso/a</option>
                                            <option>Taglia comoda</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: '#94a3b8', fontSize: '10px' }}>▼</div>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Caratteristiche Fisiche e Abbigliamento */}
                            <div style={{ marginBottom: '16px' }}>
                                <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Dettagli Fisici e Abbigliamento</label>
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
                                    placeholder="Es: Occhiali eleganti, orecchini, cappello, lentiggini, tatuaggi..."
                                    value={config.physicalTraits}
                                    onChange={(e) => setConfig({...config, physicalTraits: e.target.value})}
                                />
                            </div>
                            
                            {/* Carattere & Personalità - Menu a tendina */}
                            <div>
                                <label style={{ display: 'block', color: '#64748b', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>Carattere & Personalità</label>
                                <div className="personality-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                    <div style={{ position: 'relative' }}>
                                        <label style={{ display: 'block', color: '#94a3b8', fontSize: '9px', fontWeight: 600, marginBottom: '4px' }}>Temperamento</label>
                                        <select
                                            style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '10px', padding: '10px 12px', fontSize: '13px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)', appearance: 'none', cursor: 'pointer' }}
                                            value={config.temperament || 'Calmo/a'}
                                            onChange={(e) => setConfig({...config, temperament: e.target.value})}
                                        >
                                            <option>Calmo/a</option>
                                            <option>Energico/a</option>
                                            <option>Riflessivo/a</option>
                                            <option>Spontaneo/a</option>
                                            <option>Paziente</option>
                                            <option>Vivace</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '10px', bottom: '12px', pointerEvents: 'none', color: '#94a3b8', fontSize: '9px' }}>▼</div>
                                    </div>
                                    <div style={{ position: 'relative' }}>
                                        <label style={{ display: 'block', color: '#94a3b8', fontSize: '9px', fontWeight: 600, marginBottom: '4px' }}>Personalità Dominante</label>
                                        <select
                                            style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '10px', padding: '10px 12px', fontSize: '13px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)', appearance: 'none', cursor: 'pointer' }}
                                            value={config.sociality || 'Empatico'}
                                            onChange={(e) => setConfig({...config, sociality: e.target.value})}
                                        >
                                            <option value="Empatico">Empatico (Caldo e premuroso)</option>
                                            <option value="Riservato">Riservato (Professionale e distaccato)</option>
                                            <option value="Introverso">Introverso (Riflessivo e profondo)</option>
                                            <option value="Estroverso">Estroverso (Energico ed entusiasta)</option>
                                            <option value="Timido">Timido (Esitante e gentile)</option>
                                            <option value="Socievole">Socievole (Amichevole e chiacchierone)</option>
                                            <option value="Selettivo">Selettivo (Sophisticated e snob)</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '10px', bottom: '12px', pointerEvents: 'none', color: '#94a3b8', fontSize: '9px' }}>▼</div>
                                    </div>
                                    <div style={{ position: 'relative' }}>
                                        <label style={{ display: 'block', color: '#94a3b8', fontSize: '9px', fontWeight: 600, marginBottom: '4px' }}>Umore</label>
                                        <select
                                            style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '10px', padding: '10px 12px', fontSize: '13px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)', appearance: 'none', cursor: 'pointer' }}
                                            value={config.mood || 'Ottimista'}
                                            onChange={(e) => setConfig({...config, mood: e.target.value})}
                                        >
                                            <option>Ottimista</option>
                                            <option>Realista</option>
                                            <option>Sognatore</option>
                                            <option>Ironico/a</option>
                                            <option>Serio/a</option>
                                            <option>Allegro/a</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '10px', bottom: '12px', pointerEvents: 'none', color: '#94a3b8', fontSize: '9px' }}>▼</div>
                                    </div>
                                    <div style={{ position: 'relative' }}>
                                        <label style={{ display: 'block', color: '#94a3b8', fontSize: '9px', fontWeight: 600, marginBottom: '4px' }}>Stile comunicativo</label>
                                        <select
                                            style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.75)', border: '1px solid rgba(226,232,240,0.6)', borderRadius: '10px', padding: '10px 12px', fontSize: '13px', color: '#1e293b', outline: 'none', fontWeight: 500, boxSizing: 'border-box', backdropFilter: 'blur(4px)', appearance: 'none', cursor: 'pointer' }}
                                            value={config.commStyle || 'Buon ascoltatore'}
                                            onChange={(e) => setConfig({...config, commStyle: e.target.value})}
                                        >
                                            <option>Buon ascoltatore</option>
                                            <option>Consigliere</option>
                                            <option>Diretto/a</option>
                                            <option>Diplomatico/a</option>
                                            <option>Incoraggiante</option>
                                            <option>Scherzoso/a</option>
                                        </select>
                                        <div style={{ position: 'absolute', right: '10px', bottom: '12px', pointerEvents: 'none', color: '#94a3b8', fontSize: '9px' }}>▼</div>
                                    </div>
                                </div>
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

                        {/* Section 5: Google Calendar (Opzionale) */}
                        <div style={{ 
                          marginBottom: '28px',
                          backgroundColor: 'rgba(34, 197, 94, 0.05)',
                          borderRadius: '16px',
                          padding: '20px',
                          border: '1px solid rgba(34, 197, 94, 0.15)'
                        }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#16a34a', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                <Calendar size={18} style={{ color: '#22c55e' }}/> Google Calendar
                                <span style={{ fontSize: '10px', fontWeight: 500, color: '#94a3b8', marginLeft: 'auto' }}>Opzionale</span>
                            </h3>
                            <p style={{ fontSize: '12px', color: '#64748b', marginBottom: '16px', lineHeight: 1.5 }}>
                                Connetti il tuo calendario per permettere al tuo confidente di ricordarti appuntamenti e impegni.
                            </p>
                            
                            {googleCalendarToken ? (
                              <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '12px 16px',
                                backgroundColor: '#f0fdf4',
                                borderRadius: '12px',
                                border: '1px solid #bbf7d0'
                              }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                  <CalendarCheck size={18} style={{ color: '#22c55e' }} />
                                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#16a34a' }}>Calendario connesso!</span>
                                </div>
                                <button
                                  onClick={disconnectGoogleCalendar}
                                  style={{
                                    padding: '6px 12px',
                                    fontSize: '11px',
                                    backgroundColor: 'transparent',
                                    color: '#64748b',
                                    border: '1px solid #e2e8f0',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    fontWeight: 500
                                  }}
                                >
                                  Disconnetti
                                </button>
                              </div>
                            ) : (
                              <button
                                onClick={initGoogleCalendar}
                                style={{
                                  width: '100%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  gap: '10px',
                                  padding: '14px 20px',
                                  backgroundColor: 'white',
                                  color: '#16a34a',
                                  borderRadius: '12px',
                                  border: '1px solid #bbf7d0',
                                  cursor: 'pointer',
                                  fontSize: '14px',
                                  fontWeight: 600,
                                  transition: 'all 0.2s',
                                  boxShadow: '0 2px 8px rgba(34, 197, 94, 0.1)'
                                }}
                              >
                                <Calendar size={18} />
                                {GOOGLE_CLIENT_ID ? "Connetti Google Calendar" : "Configura Calendar (VITE_GOOGLE_CLIENT_ID)"}
                              </button>
                            )}
                        </div>

                        {/* PULSANTE CREAZIONE - Alla fine del form */}
                        <button
                          onClick={isFormComplete && !isGeneratingProfile ? handleConfigSubmit : undefined}
                          disabled={!isFormComplete || isGeneratingProfile}
                          style={{
                            width: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '16px',
                            padding: '20px 24px',
                            borderRadius: '16px',
                            backgroundColor: isFormComplete ? '#9333ea' : '#e2e8f0',
                            border: 'none',
                            transition: 'all 0.3s ease',
                            boxShadow: isFormComplete ? '0 8px 24px rgba(147, 51, 234, 0.3)' : 'none',
                            cursor: isFormComplete && !isGeneratingProfile ? 'pointer' : 'not-allowed',
                            outline: 'none',
                            marginTop: '8px'
                          }}>
                            <div style={{
                              width: '36px',
                              height: '36px',
                              borderRadius: '50%',
                              backgroundColor: 'rgba(255,255,255,0.2)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white',
                              animation: isFormComplete && !isGeneratingProfile ? 'heartPulse 1s ease-in-out infinite' : 'none'
                            }}>
                                {isGeneratingProfile ? <Loader2 className="animate-spin" size={20} /> : <Heart fill="currentColor" size={20} />}
                            </div>
                            <span style={{ fontWeight: 700, color: 'white', fontSize: '16px' }}>
                              {isGeneratingProfile ? (loadingStep || 'Creazione in corso...') : (isFormComplete ? 'Crea il tuo Confidente' : 'Compila tutti i campi')}
                            </span>
                        </button>
                        
                        {/* Copyright - Alla fine */}
                        <div style={{
                          marginTop: '24px',
                          textAlign: 'center',
                          fontSize: '10px',
                          fontWeight: 500,
                          color: '#94a3b8',
                          letterSpacing: '0.05em',
                          paddingBottom: '20px'
                        }}>
                            © Copyright Effetre Properties IA Division 2025 - All rights reserved
                        </div>
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
      backgroundImage: "url('/background.png')",
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundColor: '#f8fafc',
      position: 'relative',
      fontFamily: 'Outfit, sans-serif',
      color: '#1e293b',
      overflow: 'hidden'
    }}>
      
      {/* CSS per mobile chat */}
      <style>{`
        @keyframes glowPulseGreen {
          0%, 100% { box-shadow: 0 0 15px 5px rgba(34, 197, 94, 0.6); }
          50% { box-shadow: 0 0 25px 10px rgba(34, 197, 94, 0.9); }
        }
        @keyframes glowPulseOrange {
          0%, 100% { box-shadow: 0 0 10px 3px rgba(249, 115, 22, 0.5); }
          50% { box-shadow: 0 0 18px 6px rgba(249, 115, 22, 0.8); }
        }
        @media (max-width: 768px) {
          .chat-sidebar {
            display: none !important;
          }
          .chat-main {
            width: 100% !important;
          }
          .desktop-visualizer {
            display: none !important;
          }
          .mobile-header-container {
            display: flex !important;
          }
          .chat-transcript {
            padding: 10px !important;
            gap: 8px !important;
          }
          .chat-transcript > div > div {
            max-width: 85% !important;
          }
          .chat-transcript > div > div > div:first-child {
            font-size: 8px !important;
          }
          .chat-transcript > div > div {
            padding: 10px 12px !important;
            font-size: 12px !important;
            line-height: 1.4 !important;
            border-radius: 12px !important;
          }
          .chat-footer {
            padding: 0 !important;
          }
        }
        @media (min-width: 769px) {
          .mobile-header-container {
            display: none !important;
          }
          .desktop-visualizer {
            display: flex !important;
          }
	  .mobile-calendar-container {
            display: none !important;
          }
        }
      `}</style>

      {/* LEFT COLUMN: PROFILE SIDEBAR - Solo desktop */}
      <aside className="chat-sidebar" style={{
        width: '380px',
        minWidth: '300px',
        maxWidth: '450px',
        backgroundColor: 'rgba(255,255,255,0.95)',
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
        
        {/* Header: Logo + Progetto Ti Ascolto - CLICCABILE per tornare al menu */}
        <div 
          onClick={() => { if(window.confirm('Vuoi tornare al menu principale? La conversazione verrà terminata.')) { resetConfiguration(); } }}
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
            <img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
          </div>
          <div>
            <div style={{ fontSize: '8px', fontWeight: 700, letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>Progetto</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', letterSpacing: '-0.02em' }}>Ti Ascolto</div>
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
          boxShadow: '0 4px 16px rgba(0,0,0,0.1)'
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
              maxHeight: '200px', // Aumentato per mostrare più testo
              overflowY: 'auto',
              margin: 0
            }}>
              "{config.biography || `Personalità: ${config.sociality}`}"
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
          
          {/* Google Calendar Connection - MODIFICATO: SEMPRE VISIBILE */}
          <div style={{ marginTop: '12px' }}>
              {googleCalendarToken && isCalendarTokenValid ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: '#f0fdf4',
                  borderRadius: '10px',
                  border: '1px solid #bbf7d0'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <CalendarCheck size={16} style={{ color: '#22c55e' }} />
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#16a34a' }}>Calendario connesso</span>
                  </div>
                  <button
                    onClick={disconnectGoogleCalendar}
                    style={{
                      padding: '4px 8px',
                      fontSize: '9px',
                      backgroundColor: 'transparent',
                      color: '#64748b',
                      border: 'none',
                      cursor: 'pointer',
                      textDecoration: 'underline'
                    }}
                  >
                    Disconnetti
                  </button>
                </div>
              ) : googleCalendarToken && isCalendarTokenValid === false ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  padding: '10px 12px',
                  backgroundColor: '#fef2f2',
                  borderRadius: '10px',
                  border: '1px solid #fecaca'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Calendar size={16} style={{ color: '#ef4444' }} />
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#dc2626' }}>Sessione scaduta</span>
                  </div>
                  <button
                    onClick={initGoogleCalendar}
                    style={{
                      width: '100%',
                      padding: '8px',
                      backgroundColor: '#ef4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    Riconnetti Calendar
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <button
                    onClick={initGoogleCalendar}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '10px',
                      padding: '14px 20px',
                      backgroundColor: 'white',
                      color: '#16a34a',
                      borderRadius: '12px',
                      border: '1px solid #bbf7d0',
                      cursor: 'pointer',
                      fontSize: '14px',
                      fontWeight: 600,
                      transition: 'all 0.2s',
                      boxShadow: '0 2px 8px rgba(34, 197, 94, 0.1)'
                    }}
                  >
                    <Calendar size={18} />
                    {GOOGLE_CLIENT_ID ? "Connetti Google Calendar" : "Configura Calendar (ID Mancante)"}
                  </button>
                  
                  {/* AVVISO UTENTE */}
                  <div style={{ 
                    fontSize: '10px', 
                    color: '#64748b', 
                    backgroundColor: 'rgba(255,255,255,0.6)', 
                    padding: '8px', 
                    borderRadius: '8px',
                    border: '1px dashed #cbd5e1'
                  }}>
                    <strong>Nota Tecnica:</strong> Se ricevi un errore "Access Blocked" o "403", invia questa email all'amministratore:
                    <br/>
                    <code style={{ 
                      display: 'block', 
                      marginTop: '4px', 
                      backgroundColor: '#f1f5f9', 
                      padding: '4px', 
                      borderRadius: '4px', 
                      fontWeight: 'bold',
                      color: '#0f172a'
                    }}>
                      {currentUser?.email || "Email non rilevata"}
                    </code>
                  </div>
                </div>
              )}
          </div>
          
          {/* RUBRICA CONTATTI */}
          <div style={{ marginTop: '12px' }}>
            <button
              onClick={() => setShowContactsModal(true)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px',
                padding: '12px 16px',
                backgroundColor: 'white',
                color: '#6366f1',
                borderRadius: '12px',
                border: '1px solid #c7d2fe',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: 600,
                transition: 'all 0.2s',
                boxShadow: '0 2px 8px rgba(99, 102, 241, 0.1)'
              }}
            >
              <Phone size={16} />
              Rubrica ({contacts.length})
            </button>
          </div>
          
          {/* Pulsante Nuovo Assistente */}
          <button
            onClick={() => { 
              if(window.confirm('Vuoi creare un nuovo assistente? La configurazione attuale verrà cancellata.')) { 
                resetConfiguration(); 
              } 
            }}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 14px',
              marginTop: '12px',
              backgroundColor: 'transparent',
              color: '#64748b',
              borderRadius: '10px',
              border: '1px dashed #cbd5e1',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#f1f5f9';
              e.currentTarget.style.borderColor = '#94a3b8';
              e.currentTarget.style.color = '#475569';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = '#cbd5e1';
              e.currentTarget.style.color = '#64748b';
            }}
          >
            <RefreshCw size={14} />
            Nuovo Assistente
          </button>
          
          {/* Pulsante Logout */}
          <button
            onClick={handleLogout}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 14px',
              marginTop: '8px',
              backgroundColor: 'transparent',
              color: '#ef4444',
              borderRadius: '10px',
              border: '1px dashed #fca5a5',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#fef2f2';
              e.currentTarget.style.borderColor = '#f87171';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = '#fca5a5';
            }}
          >
            <LogOut size={14} />
            Esci
          </button>
          
          {/* User info */}
          {currentUser && (
            <div style={{
              marginTop: '12px',
              padding: '8px 12px',
              backgroundColor: 'rgba(147, 51, 234, 0.05)',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <p style={{ 
                fontSize: '10px', 
                color: '#64748b', 
                margin: 0,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}>
                {currentUser.email}
              </p>
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
      <main className="chat-main" style={{
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
            fontSize: '14px',
            maxWidth: '90%',
            textAlign: 'center'
          }}>
            <Info size={18} /> {error}
          </div>
        )}

        {/* MOBILE HEADER - Layout completo con foto, info e logo */}
        <div 
          className="mobile-header-container"
          style={{
            display: 'none', // Gestito da CSS
            flexDirection: 'column',
            backgroundColor: 'rgba(255,255,255,0.98)',
            backdropFilter: 'blur(12px)',
            borderBottom: '1px solid rgba(226,232,240,0.6)',
            flexShrink: 0
          }}
        >
          {/* Riga principale: Foto + Info + Logo */}
          <div style={{ display: 'flex', padding: '12px', gap: '12px', alignItems: 'stretch' }}>
            {/* Foto rettangolare con alone colorato - DIMENSIONI AUMENTATE */}
            <div 
              onClick={() => { if(window.confirm('Vuoi tornare al menu principale?')) { resetConfiguration(); } }}
              style={{
                width: '140px', // Aumentato da 100px
                aspectRatio: '3/4', // Mantiene la proporzione
                borderRadius: '8px',
                overflow: 'hidden',
                flexShrink: 0,
                cursor: 'pointer',
                boxShadow: !isConnected 
                  ? '0 2px 12px rgba(0,0,0,0.1)'
                  : isMuted 
                    ? '0 0 12px 4px rgba(239, 68, 68, 0.7)'
                    : audioVolume > 0.1 
                      ? '0 0 15px 6px rgba(34, 197, 94, 0.8)'
                      : '0 0 10px 4px rgba(249, 115, 22, 0.6)',
                animation: !isConnected || isMuted 
                  ? 'none' 
                  : audioVolume > 0.1 
                    ? 'glowPulseGreen 1s ease-in-out infinite' 
                    : 'glowPulseOrange 1.5s ease-in-out infinite',
                transition: 'box-shadow 0.3s ease',
                position: 'relative' // Per posizionare elementi interni se necessario
              }}
            >
              {avatarUrl ? (
                <img src={avatarUrl} alt="Avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              ) : (
                <div style={{ width: '100%', height: '100%', backgroundColor: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <UserCircle size={40} style={{ color: '#cbd5e1' }} />
                </div>
              )}
            </div>
            
            {/* Info centrale - Si allunga per matchare l'altezza della foto */}
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                {/* Nome */}
                <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#0f172a', marginBottom: '2px', lineHeight: 1.2 }}>
                  {config.name || 'Il tuo Confidente'}
                </h2>
                <p style={{ fontSize: '11px', color: '#64748b', marginBottom: '8px' }}>
                  Confidente di {config.userName}
                </p>
                
                {/* Età */}
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginBottom: '6px' }}>
                  <span>ETÀ</span>
                  <span style={{ color: '#0f172a', fontWeight: 600 }}>{config.age} anni</span>
                </div>
                
                {/* Biografia label */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
                  <Heart size={10} fill="#ec4899" style={{ color: '#ec4899' }} />
                  <span style={{ fontSize: '9px', color: '#ec4899', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Biografia</span>
                </div>
              </div>
              
              {/* Testo biografia - Occupa lo spazio rimanente e si allinea in basso */}
              <div style={{ 
                flex: 1,
                display: 'flex',
                alignItems: 'flex-end'
              }}>
                <p style={{ 
                  fontSize: '10px', 
                  color: '#475569', 
                  lineHeight: 1.4, 
                  maxHeight: '80px', // Altezza massima scrollabile se necessario
                  overflowY: 'auto',
                  margin: 0
                }}>
                  {config.biography || 'Nessuna biografia disponibile.'}
                </p>
              </div>
            </div>
            
            {/* Logo Ti Ascolto - In alto a destra */}
            <div style={{ position: 'absolute', top: '12px', right: '12px', display: 'flex', flexDirection: 'column', alignItems: 'center', opacity: 0.9, zIndex: 10 }}>
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                overflow: 'hidden',
                backgroundColor: 'white',
                boxShadow: '0 2px 8px rgba(147, 112, 219, 0.15)'
              }}>
                <img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
              </div>
            </div>
          </div>
        </div>

        {/* Central Visualizer Area - SOLO DESKTOP */}
        <div className="desktop-visualizer" style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '16px',
          padding: '10px 20px',
          flexShrink: 0,
          borderBottom: '1px solid rgba(226,232,240,0.4)',
          backgroundColor: 'rgba(255,255,255,0.4)'
        }}>
          <div className="visualizer-wrapper" style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', transform: 'scale(0.5)' }}>
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
          className="chat-transcript"
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
                    <button 
                      onClick={() => {
                        // Per mailto: usiamo location.href, per altri link window.open
                        if (t.actionUrl?.startsWith('mailto:')) {
                          window.location.href = t.actionUrl;
                        } else {
                          window.open(t.actionUrl, '_blank', 'noopener,noreferrer');
                        }
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        padding: '16px',
                        borderRadius: '12px',
                        fontWeight: 700,
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        textAlign: 'left',
                        width: '100%',
                        background: t.actionIcon === 'mail' 
                          ? 'linear-gradient(135deg, #ec4899, #f43f5e)' 
                          : t.actionIcon === 'send'
                            ? 'linear-gradient(135deg, #0088cc, #00a0dc)'
                            : t.actionIcon === 'phone'
                              ? 'linear-gradient(135deg, #f59e0b, #d97706)'
                              : 'linear-gradient(135deg, #10b981, #14b8a6)',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.15)'
                      }}
                    >
                      <div style={{ padding: '8px', backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: '50%' }}>
                        {t.actionIcon === 'mail' ? <Mail size={20} /> : t.actionIcon === 'send' ? <Send size={20} /> : t.actionIcon === 'phone' ? <Phone size={20} /> : <MessageCircle size={20} />}
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '15px' }}>{t.actionLabel}</span>
                        <span style={{ fontSize: '10px', opacity: 0.8, fontWeight: 400 }}>Tocca per aprire</span>
                      </div>
                      <ExternalLink size={16} style={{ marginLeft: 'auto', opacity: 0.8 }} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Footer Area con pulsanti e copyright */}
        <div className="chat-footer" style={{
          borderTop: '1px solid rgba(226,232,240,0.5)',
          backgroundColor: 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(8px)',
          flexShrink: 0
        }}>
{/* --- NUOVO BLOCCO CALENDARIO PER MOBILE --- */}
          <div className="mobile-calendar-container" style={{ padding: '10px 16px 0 16px' }}>
              {googleCalendarToken && isCalendarTokenValid ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  backgroundColor: '#f0fdf4',
                  borderRadius: '10px',
                  border: '1px solid #bbf7d0'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <CalendarCheck size={14} style={{ color: '#22c55e' }} />
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#16a34a' }}>Calendario connesso</span>
                  </div>
                  <button
                    onClick={disconnectGoogleCalendar}
                    style={{
                      padding: '4px 8px',
                      fontSize: '10px',
                      backgroundColor: 'transparent',
                      color: '#64748b',
                      border: 'none',
                      cursor: 'pointer',
                      textDecoration: 'underline'
                    }}
                  >
                    Disconnetti
                  </button>
                </div>
              ) : googleCalendarToken && isCalendarTokenValid === false ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                  padding: '8px 12px',
                  backgroundColor: '#fef2f2',
                  borderRadius: '10px',
                  border: '1px solid #fecaca'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Calendar size={14} style={{ color: '#ef4444' }} />
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#dc2626' }}>Sessione scaduta</span>
                  </div>
                  <button
                    onClick={initGoogleCalendar}
                    style={{
                      width: '100%',
                      padding: '6px',
                      backgroundColor: '#ef4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      fontSize: '10px',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    Riconnetti
                  </button>
                </div>
              ) : (
                <button
                  onClick={initGoogleCalendar}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    padding: '10px 14px',
                    backgroundColor: '#f3e8ff',
                    color: '#7e22ce',
                    borderRadius: '10px',
                    border: '1px solid #d8b4fe',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: 600,
                    transition: 'all 0.2s'
                  }}
                >
                  <Calendar size={16} />
                  {GOOGLE_CLIENT_ID ? "Connetti Google Calendar" : "Configura Calendar"}
                </button>
              )}
          </div>
          
          {/* Pulsante Rubrica Mobile */}
          <div style={{ padding: '8px 16px 0 16px' }}>
            <button
              onClick={() => setShowContactsModal(true)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '10px 14px',
                backgroundColor: '#eef2ff',
                color: '#6366f1',
                borderRadius: '10px',
                border: '1px solid #c7d2fe',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 600
              }}
            >
              <Phone size={14} />
              Rubrica ({contacts.length})
            </button>
          </div>
          
          {/* Pulsante Nuovo Assistente Mobile */}
          <div style={{ padding: '8px 16px 0 16px' }}>
            <button
              onClick={() => { 
                if(window.confirm('Vuoi creare un nuovo assistente? La configurazione attuale verrà cancellata.')) { 
                  resetConfiguration(); 
                } 
              }}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '10px 14px',
                backgroundColor: 'transparent',
                color: '#64748b',
                borderRadius: '10px',
                border: '1px dashed #cbd5e1',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 500
              }}
            >
              <RefreshCw size={14} />
              Nuovo Assistente
            </button>
            
            {/* Pulsante Logout Mobile */}
            <button
              onClick={handleLogout}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '10px 14px',
                marginTop: '8px',
                backgroundColor: 'transparent',
                color: '#ef4444',
                borderRadius: '10px',
                border: '1px dashed #fca5a5',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 500
              }}
            >
              <LogOut size={14} />
              Esci ({currentUser?.email?.split('@')[0]})
            </button>
          </div>
          {/* --- FINE NUOVO BLOCCO --- */}          
	{/* Riga pulsanti */}
          <div style={{
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <input 
              type="file" 
              ref={fileInputRef}
              accept="image/*"
              onChange={handleUserPhotoUpload}
              style={{ display: 'none' }}
            />
            
            {/* Pulsante Invia foto */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzingPhoto}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '12px 16px',
                backgroundColor: 'white',
                color: isAnalyzingPhoto ? '#94a3b8' : '#64748b',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: isAnalyzingPhoto ? 'not-allowed' : 'pointer',
                flexShrink: 0
              }}
            >
              {isAnalyzingPhoto ? <Loader2 size={14} className="animate-spin" /> : <Camera size={14} />}
              <span>Invia una foto</span>
            </button>
            
            {/* Pulsante principale: Connect/Disconnect/Mute */}
            {!isConnected ? (
              <button 
                onClick={connect}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  padding: '12px 20px',
                  backgroundColor: '#0f172a',
                  color: 'white',
                  borderRadius: '10px',
                  fontWeight: 700,
                  fontSize: '13px',
                  border: 'none',
                  cursor: 'pointer',
                  boxShadow: '0 2px 8px rgba(15, 23, 42, 0.2)'
                }}
              >
                <Mic size={16} />
                INIZIA A PARLARE
              </button>
            ) : (
              <div style={{ flex: 1, display: 'flex', gap: '8px' }}>
                <button 
                  onClick={toggleMute}
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '12px',
                    backgroundColor: isMuted ? '#fef2f2' : 'white',
                    color: isMuted ? '#ef4444' : '#22c55e',
                    borderRadius: '10px',
                    fontWeight: 600,
                    border: isMuted ? '1px solid #fecaca' : '1px solid #bbf7d0',
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
                    fontSize: '12px',
                    border: '1px solid #fecaca',
                    cursor: 'pointer'
                  }}
                >
                  <PhoneOff size={16} />
                  Termina
                </button>
              </div>
            )}
          </div>
          
          {/* Didascalia foto */}
          <div style={{ 
            textAlign: 'center', 
            fontSize: '10px', 
            color: '#94a3b8', 
            paddingBottom: '8px' 
          }}>
            Condividi un'immagine con {config.name}
          </div>
          
          {/* Copyright */}
          <div style={{
            padding: '10px 16px',
            textAlign: 'center',
            fontSize: '8px',
            fontWeight: 500,
            color: '#94a3b8',
            borderTop: '1px solid rgba(226,232,240,0.3)',
            letterSpacing: '0.02em'
          }}>
            © Effetre Properties IA Division 2025 - All rights reserved
          </div>
        </div>
      </main>
      
      {/* MODAL RUBRICA */}
      {showContactsModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '20px'
        }} onClick={() => setShowContactsModal(false)}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '16px',
            padding: '24px',
            maxWidth: '400px',
            width: '100%',
            maxHeight: '80vh',
            overflow: 'auto',
            boxShadow: '0 20px 40px rgba(0,0,0,0.2)'
          }} onClick={e => e.stopPropagation()}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: 700, color: '#1e293b' }}>
              📒 Rubrica Contatti
            </h3>
            
            {/* Pulsanti Importazione */}
            <div style={{ 
              display: 'flex', 
              gap: '8px', 
              marginBottom: '16px',
              padding: '12px',
              backgroundColor: '#eef2ff',
              borderRadius: '10px',
              border: '1px solid #c7d2fe'
            }}>
              <button
                onClick={importContactsFromPhone}
                style={{
                  flex: 1,
                  padding: '10px 8px',
                  backgroundColor: '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px'
                }}
              >
                <Phone size={14} />
                Importa dal Telefono
              </button>
              
              <label style={{
                flex: 1,
                padding: '10px 8px',
                backgroundColor: 'white',
                color: '#6366f1',
                border: '1px solid #c7d2fe',
                borderRadius: '8px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px'
              }}>
                <Download size={14} />
                Importa .vcf
                <input
                  type="file"
                  accept=".vcf,.vcard"
                  onChange={handleVcfImport}
                  style={{ display: 'none' }}
                />
              </label>
            </div>
            
            <p style={{ 
              fontSize: '10px', 
              color: '#64748b', 
              margin: '0 0 16px 0',
              textAlign: 'center'
            }}>
              💡 Su Android/Chrome puoi importare direttamente. Su iOS/Safari esporta i contatti come file .vcf
            </p>
            
            {/* Form per aggiungere contatto */}
            <form onSubmit={(e) => {
              e.preventDefault();
              const form = e.target as HTMLFormElement;
              const name = (form.elements.namedItem('contactName') as HTMLInputElement).value;
              const phone = (form.elements.namedItem('contactPhone') as HTMLInputElement).value;
              const telegram = (form.elements.namedItem('contactTelegram') as HTMLInputElement).value;
              if (name && phone) {
                addContact(name, phone, telegram || undefined);
                form.reset();
              }
            }} style={{ marginBottom: '20px', padding: '16px', backgroundColor: '#f8fafc', borderRadius: '12px' }}>
              <div style={{ marginBottom: '12px' }}>
                <input
                  name="contactName"
                  type="text"
                  placeholder="Nome *"
                  required
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>
              <div style={{ marginBottom: '12px' }}>
                <input
                  name="contactPhone"
                  type="tel"
                  placeholder="Telefono (es. +393331234567) *"
                  required
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>
              <div style={{ marginBottom: '12px' }}>
                <input
                  name="contactTelegram"
                  type="text"
                  placeholder="Username Telegram (opzionale)"
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>
              <button
                type="submit"
                style={{
                  width: '100%',
                  padding: '10px',
                  backgroundColor: '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                + Aggiungi Contatto
              </button>
            </form>
            
            {/* Lista contatti */}
            <div style={{ maxHeight: '300px', overflow: 'auto' }}>
              {contacts.length === 0 ? (
                <p style={{ textAlign: 'center', color: '#94a3b8', fontSize: '13px' }}>
                  Nessun contatto. Aggiungi contatti per poter dire "Chiama Marco" all'assistente!
                </p>
              ) : (
                contacts.map(contact => (
                  <div key={contact.id} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px',
                    borderBottom: '1px solid #f1f5f9'
                  }}>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '14px', color: '#1e293b' }}>{contact.name}</div>
                      <div style={{ fontSize: '12px', color: '#64748b' }}>{contact.phone}</div>
                      {contact.telegram && (
                        <div style={{ fontSize: '11px', color: '#0ea5e9' }}>@{contact.telegram}</div>
                      )}
                    </div>
                    <button
                      onClick={() => removeContact(contact.id)}
                      style={{
                        padding: '6px 12px',
                        backgroundColor: '#fef2f2',
                        color: '#ef4444',
                        border: 'none',
                        borderRadius: '6px',
                        fontSize: '11px',
                        cursor: 'pointer'
                      }}
                    >
                      Elimina
                    </button>
                  </div>
                ))
              )}
            </div>
            
            {/* Pulsanti azione */}
            <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
              {contacts.length > 0 && (
                <button
                  onClick={() => {
                    if (window.confirm(`Vuoi eliminare tutti i ${contacts.length} contatti?`)) {
                      setContacts([]);
                      localStorage.removeItem('ti_ascolto_contacts');
                    }
                  }}
                  style={{
                    flex: 1,
                    padding: '12px',
                    backgroundColor: '#fef2f2',
                    color: '#ef4444',
                    border: '1px solid #fecaca',
                    borderRadius: '8px',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer'
                  }}
                >
                  Elimina Tutti ({contacts.length})
                </button>
              )}
              <button
                onClick={() => setShowContactsModal(false)}
                style={{
                  flex: contacts.length > 0 ? 1 : undefined,
                  width: contacts.length > 0 ? undefined : '100%',
                  padding: '12px',
                  backgroundColor: '#f1f5f9',
                  color: '#64748b',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 500,
                  cursor: 'pointer'
                }}
              >
                Chiudi
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;