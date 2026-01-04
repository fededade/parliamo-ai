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
      end_datetime: { type: Type.STRING, description: