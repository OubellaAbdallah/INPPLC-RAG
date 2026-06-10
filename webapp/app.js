/* =========================================================
   INPPLC Chatbot — app.js
   Handles: sessions, messaging, sidebar, search, SSE streaming
   ========================================================= */

'use strict';

// ── API base ──────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// ── State ─────────────────────────────────────────────────
const state = {
  currentSessionId: null,
  sessions: [],          // [{session_id, preview}]
  messages: [],          // [{role, content}]
  isStreaming: false,
  sidebarOpen: true,
  searchQuery: '',
};

// ── DOM refs ──────────────────────────────────────────────
const sidebar         = document.getElementById('sidebar');
const sidebarToggleBtn= document.getElementById('sidebarToggleBtn');
const newChatBtn      = document.getElementById('newChatBtn');
const recentChatsBtn  = document.getElementById('recentChatsBtn');
const sessionsList    = document.getElementById('sessionsList');
const sidebarSearch   = document.getElementById('sidebarSearch');
const chatWelcome     = document.getElementById('chatWelcome');
const messagesFeed    = document.getElementById('messagesFeed');
const messageInput    = document.getElementById('messageInput');
const sendBtn         = document.getElementById('sendBtn');
const clearChatBtn    = document.getElementById('clearChatBtn');
const topbarSearch    = document.getElementById('topbarSearch');
const floatingSearchBtn = document.getElementById('floatingSearchBtn');
const warningBtn      = document.getElementById('warningBtn');
const auditTrailPanel = document.getElementById('auditTrailPanel');
const auditPanelCloseBtn = document.getElementById('auditPanelCloseBtn');
const auditActiveQuery = document.getElementById('auditActiveQuery');
const auditSourcesList = document.getElementById('auditSourcesList');

// INPPLC logo avatar URL
const BOT_AVATAR = 'assets/logo.png';
// Generic user avatar – initials fallback
const USER_AVATAR = null;

// ── Utility: escape HTML ──────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Utility: toast notification ───────────────────────────
function showToast(msg, duration = 2400) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateY(12px)';
    t.style.transition = 'opacity .3s, transform .3s';
    setTimeout(() => t.remove(), 350);
  }, duration);
}

// ── Auto-resize textarea ──────────────────────────────────
messageInput.addEventListener('input', () => {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + 'px';
  sendBtn.disabled = messageInput.value.trim() === '' || state.isStreaming;
});

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) handleSend();
  }
});

// ── Sidebar toggle ────────────────────────────────────────
function openSidebar() {
  const isMobile = window.innerWidth <= 768;
  if (isMobile) {
    sidebar.classList.add('mobile-open');
  } else {
    sidebar.classList.remove('collapsed');
  }
  document.body.classList.remove('sidebar-closed');
  sidebarToggleBtn.setAttribute('aria-label', 'Close sidebar');
  state.sidebarOpen = true;
}

function closeSidebar() {
  const isMobile = window.innerWidth <= 768;
  if (isMobile) {
    sidebar.classList.remove('mobile-open');
  } else {
    sidebar.classList.add('collapsed');
  }
  document.body.classList.add('sidebar-closed');
  sidebarToggleBtn.setAttribute('aria-label', 'Open sidebar');
  state.sidebarOpen = false;
}

sidebarToggleBtn.addEventListener('click', () => {
  state.sidebarOpen ? closeSidebar() : openSidebar();
});

// ── Sessions API ──────────────────────────────────────────
async function fetchSessions() {
  try {
    const res = await fetch(`${API_BASE}/api/sessions`);
    const data = await res.json();
    state.sessions = data.sessions || [];
    renderSessionsList();
  } catch {
    // Offline / dev mode — ignore
  }
}

async function createSession() {
  try {
    const res = await fetch(`${API_BASE}/api/session`, { method: 'POST' });
    const data = await res.json();
    return data.session_id;
  } catch {
    // Fallback: generate local UUID
    return crypto.randomUUID();
  }
}

async function deleteSession(sessionId) {
  try {
    await fetch(`${API_BASE}/api/session/${sessionId}`, { method: 'DELETE' });
  } catch { /* ignore */ }
  state.sessions = state.sessions.filter(s => s.session_id !== sessionId);
  if (state.currentSessionId === sessionId) {
    startNewChat();
  } else {
    renderSessionsList();
  }
}

async function loadSessionHistory(sessionId) {
  try {
    const res = await fetch(`${API_BASE}/api/session/${sessionId}/history`);
    const data = await res.json();
    return data.messages || [];
  } catch {
    return [];
  }
}

// ── Render sessions sidebar ───────────────────────────────
function renderSessionsList(filter = '') {
  sessionsList.innerHTML = '';
  const query = filter.toLowerCase();
  const filtered = state.sessions.filter(s =>
    !query || (s.preview || '').toLowerCase().includes(query)
  );

  if (filtered.length === 0) {
    const empty = document.createElement('div');
    empty.style.cssText = 'color:rgba(255,255,255,.45);font-size:.8rem;text-align:center;padding:16px 8px;';
    empty.textContent = filter ? 'No matching chats' : 'No recent chats';
    sessionsList.appendChild(empty);
    return;
  }

  filtered.forEach(sess => {
    const item = document.createElement('div');
    item.className = 'session-item' + (sess.session_id === state.currentSessionId ? ' active' : '');
    item.dataset.id = sess.session_id;

    const label = document.createElement('span');
    label.className = 'session-item__label';
    label.textContent = sess.preview || 'Chat';
    if (filter) label.innerHTML = highlightText(escapeHtml(label.textContent), filter);

    const delBtn = document.createElement('button');
    delBtn.className = 'session-item__delete';
    delBtn.title = 'Delete chat';
    delBtn.setAttribute('aria-label', 'Delete chat');
    delBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>`;

    item.appendChild(delBtn);
    item.appendChild(label);

    item.addEventListener('click', (e) => {
      if (e.target.closest('.session-item__delete')) return;
      switchToSession(sess.session_id, sess.preview);
    });

    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(sess.session_id);
    });

    sessionsList.appendChild(item);
  });
}

function highlightText(html, query) {
  if (!query) return html;
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  return html.replace(regex, '<mark class="highlight">$1</mark>');
}

// ── Chat: new / switch ────────────────────────────────────
async function startNewChat() {
  const sid = await createSession();
  state.currentSessionId = sid;
  state.messages = [];
  messagesFeed.innerHTML = '';
  chatWelcome.classList.remove('hidden');
  messagesFeed.style.display = 'none';

  // Add to sessions list
// Add to sessions list
  if (!state.sessions.find(s => s.session_id === sid)) {
    state.sessions.unshift({ session_id: sid, preview: 'New Chat' });
  }
  renderSessionsList();
}

// ── Message actions and Audit Trail ──────────────────────────
function createMessageMeta(role, content, sources, idx) {
  const meta = document.createElement('div');
  meta.className = 'msg-meta';

  // Copy button
  const copyBtn = document.createElement('button');
  copyBtn.className = 'msg-action-btn';
  copyBtn.title = 'Copier le message';
  copyBtn.setAttribute('aria-label', 'Copier le message');
  copyBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </svg>
    <span>Copier</span>
  `;
  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(content).then(() => showToast('Copié !', 1600));
  });
  meta.appendChild(copyBtn);

  if (role === 'assistant') {
    // Sources button
    if (sources && sources.length > 0) {
      const sourcesBtn = document.createElement('button');
      sourcesBtn.className = 'msg-action-btn msg-sources-btn';
      sourcesBtn.title = 'Traçabilité des sources';
      sourcesBtn.setAttribute('aria-label', 'Traçabilité des sources');
      sourcesBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
        </svg>
        <span>Sources (${sources.length})</span>
      `;
      sourcesBtn.addEventListener('click', () => {
        toggleAuditTrail(idx);
      });
      meta.appendChild(sourcesBtn);
    }

    // Export PDF button
    const pdfBtn = document.createElement('button');
    pdfBtn.className = 'msg-action-btn msg-pdf-btn';
    pdfBtn.title = 'Exporter en PDF';
    pdfBtn.setAttribute('aria-label', 'Exporter en PDF');
    pdfBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
      <span>PDF</span>
    `;
    pdfBtn.addEventListener('click', () => {
      exportToPDF(idx);
    });
    meta.appendChild(pdfBtn);
  }

  return meta;
}

function toggleAuditTrail(idx) {
  if (!auditTrailPanel || !auditActiveQuery || !auditSourcesList) return;

  const msg = state.messages[idx];
  if (!msg || msg.role !== 'assistant') return;

  // Retrieve user query
  let userQuery = "Requête non spécifiée";
  if (idx > 0 && state.messages[idx - 1] && state.messages[idx - 1].role === 'user') {
    userQuery = state.messages[idx - 1].content;
  }

  // Populate Active Query
  auditActiveQuery.innerHTML = `Requête de consultation :<br><strong>"${userQuery}"</strong>`;

  // Populate Sources List
  auditSourcesList.innerHTML = '';
  if (!msg.sources || msg.sources.length === 0) {
    auditSourcesList.innerHTML = '<div style="color: var(--grey-500); font-size: 0.8rem;">Aucune source documentée pour cette réponse.</div>';
  } else {
    msg.sources.forEach((src, sIdx) => {
      const card = document.createElement('div');
      card.className = 'source-card';

      const pertinence = Math.max(0, Math.min(100, Math.round((1 - src.score) * 100)));
      card.innerHTML = `
        <div class="source-card-header">
          <div class="source-card-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
            <span class="source-file-name">Source ${sIdx + 1}</span>
          </div>
          <span class="source-score">${pertinence}% pertinence</span>
        </div>
        <div class="source-card-excerpt">${src.content.substring(0, 300)}...</div>
        <button class="source-view-full-btn">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
          <span>Voir le texte complet</span>
        </button>
        <div class="source-card-full-text hidden">${src.content}</div>
      `;

      const toggleBtn = card.querySelector('.source-view-full-btn');
      const fullTextDiv = card.querySelector('.source-card-full-text');
      toggleBtn.addEventListener('click', () => {
        const isHidden = fullTextDiv.classList.toggle('hidden');
        toggleBtn.classList.toggle('expanded', !isHidden);
        toggleBtn.querySelector('span').textContent = isHidden ? 'Voir le texte complet' : 'Masquer le texte complet';
      });

      auditSourcesList.appendChild(card);
    });
  }

  // Open the panel
  auditTrailPanel.classList.remove('collapsed');
}

function exportToPDF(idx) {
  const msg = state.messages[idx];
  if (!msg || msg.role !== 'assistant') return;

  let userQuery = "Non spécifiée";
  if (idx > 0 && state.messages[idx - 1] && state.messages[idx - 1].role === 'user') {
    userQuery = state.messages[idx - 1].content;
  }

  const container = document.createElement('div');
  container.className = 'pdf-export-container';
  container.style.padding = '40px';
  container.style.fontFamily = "'Inter', Arial, sans-serif";
  container.style.color = '#333';
  container.style.background = '#fff';
  container.style.lineHeight = '1.6';

  const today = new Date().toLocaleDateString('fr-FR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  container.innerHTML = `
    <div style="border-bottom: 2px solid #1B5E35; padding-bottom: 15px; margin-bottom: 25px; display: flex; align-items: center; justify-content: space-between;">
      <div>
        <h2 style="margin: 0; color: #1B5E35; font-size: 1.6rem; text-transform: uppercase; letter-spacing: 0.5px;">INPPLC</h2>
        <p style="margin: 3px 0 0; color: #666; font-size: 0.85rem; font-weight: 500;">Assistant d'Information Juridique Multimodal</p>
      </div>
      <div style="text-align: right;">
        <span style="font-size: 0.75rem; background: #C8A84B; color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; text-transform: uppercase;">Consultation RAG</span>
      </div>
    </div>

    <h1 style="text-align: center; color: #1B5E35; font-size: 1.4rem; margin-bottom: 30px; font-weight: 700; border-bottom: 1px solid #eee; padding-bottom: 10px;">
      FICHE DE CONSULTATION JURIDIQUE ET D'ORIENTATION
    </h1>

    <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px; font-size: 0.85rem; background: #f9f9f9; border: 1px solid #eee;">
      <tr>
        <td style="padding: 10px; border: 1px solid #eee; font-weight: bold; width: 25%; color: #1B5E35;">Date de génération</td>
        <td style="padding: 10px; border: 1px solid #eee; width: 75%;">${today}</td>
      </tr>
      <tr>
        <td style="padding: 10px; border: 1px solid #eee; font-weight: bold; color: #1B5E35;">Référence Session</td>
        <td style="padding: 10px; border: 1px solid #eee; font-family: monospace;">${state.currentSessionId || 'N/A'}</td>
      </tr>
      <tr>
        <td style="padding: 10px; border: 1px solid #eee; font-weight: bold; color: #1B5E35;">Nature du document</td>
        <td style="padding: 10px; border: 1px solid #eee;">Note d'orientation et de traçabilité juridique</td>
      </tr>
    </table>

    <div style="margin-bottom: 30px;">
      <h3 style="color: #1B5E35; border-bottom: 1px solid #1B5E35; padding-bottom: 5px; margin-top: 0; font-size: 1.1rem;">
        1. Situation / Faits Signalés
      </h3>
      <p style="background: #fdfaf2; border-left: 3px solid #C8A84B; padding: 12px 15px; margin: 10px 0; font-style: italic; font-size: 0.95rem; white-space: pre-wrap;">
        "${userQuery}"
      </p>
    </div>

    <div style="margin-bottom: 30px; page-break-inside: avoid;">
      <h3 style="color: #1B5E35; border-bottom: 1px solid #1B5E35; padding-bottom: 5px; margin-top: 0; font-size: 1.1rem;">
        2. Analyse et Consultation Juridique
      </h3>
      <div style="font-size: 0.95rem; line-height: 1.6; text-align: justify; white-space: pre-wrap;">
        ${msg.content}
      </div>
    </div>

    <div style="margin-bottom: 30px; page-break-inside: avoid;">
      <h3 style="color: #1B5E35; border-bottom: 1px solid #1B5E35; padding-bottom: 5px; margin-top: 0; font-size: 1.1rem;">
        3. Recommandations et Actions à Prendre
      </h3>
      <ul style="padding-left: 20px; font-size: 0.9rem; line-height: 1.6;">
        <li style="margin-bottom: 8px;"><strong>Vérification des Sources :</strong> Veuillez croiser les extraits ci-dessous avec les publications officielles du Bulletin Officiel du Royaume du Maroc.</li>
        <li style="margin-bottom: 8px;"><strong>Voie de Dénonciation :</strong> Si les faits signalés constituent des délits de corruption avérés, ils peuvent être déclarés sur le portail officiel de dénonciation de l'INPPLC.</li>
        <li style="margin-bottom: 8px;"><strong>Assistance Institutionnelle :</strong> Pour toute démarche juridique officielle ou dépôt formel de plainte, prenez contact directement avec les services d'accueil et d'orientation de l'INPPLC.</li>
      </ul>
    </div>

    ${msg.sources && msg.sources.length > 0 ? `
    <div style="margin-bottom: 20px; page-break-inside: avoid;">
      <h3 style="color: #1B5E35; border-bottom: 1px solid #1B5E35; padding-bottom: 5px; margin-top: 0; font-size: 1.1rem;">
        4. Traçabilité des Textes de Loi Applicables
      </h3>
      <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 10px;">
        ${msg.sources.map((src, sIdx) => {
          const pertinence = Math.max(0, Math.min(100, Math.round((1 - src.score) * 100)));
          return `
          <div style="border: 1px solid #ddd; border-radius: 4px; padding: 12px; margin-bottom: 15px; page-break-inside: avoid; background: #fafafa;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 6px;">
              <span style="font-weight: bold; color: #1B5E35; font-size: 0.85rem;">Source ${sIdx + 1}</span>
              <span style="font-size: 0.75rem; background: #e2f0d9; color: #385723; padding: 2px 6px; border-radius: 10px; font-weight: bold;">Score de pertinence : ${pertinence}%</span>
            </div>
            <p style="margin: 0; font-size: 0.8rem; font-family: monospace; color: #444; white-space: pre-wrap; line-height: 1.5; background: #fff; padding: 8px; border: 1px solid #e2e2e2;">
              ${src.content}
            </p>
          </div>
          `;
        }).join('')}
      </div>
    </div>
    ` : ''}

    <div style="margin-top: 50px; border-top: 1px solid #eee; padding-top: 15px; text-align: center; font-size: 0.75rem; color: #888; page-break-inside: avoid;">
      <p style="margin: 0;">Ce rapport est fourni à titre indicatif par l'Assistant Intelligent de l'INPPLC.</p>
      <p style="margin: 3px 0 0;">Instance Nationale de la Probité, de la Prévention et de la Lutte contre la Corruption - Royaume du Maroc.</p>
    </div>
  `;

  const opt = {
    margin:       10,
    filename:     `INPPLC_Fiche_Consultation_${new Date().toISOString().slice(0,10)}.pdf`,
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true },
    jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };

  html2pdf().set(opt).from(container).save().then(() => {
    showToast('Fiche PDF téléchargée !', 2000);
  }).catch(err => {
    console.error('PDF export failed:', err);
    showToast('Échec de l\'export PDF', 2000);
  });
}

async function switchToSession(sessionId, preview) {
  if (auditTrailPanel) auditTrailPanel.classList.add('collapsed');

  state.currentSessionId = sessionId;
  const msgs = await loadSessionHistory(sessionId);
  state.messages = msgs;

  messagesFeed.innerHTML = '';
  if (msgs.length === 0) {
    chatWelcome.classList.remove('hidden');
    messagesFeed.style.display = 'none';
  } else {
    chatWelcome.classList.add('hidden');
    messagesFeed.style.display = 'flex';
    msgs.forEach((m, idx) => appendMessage(m.role, m.content, false, m.sources, idx));
    scrollToBottom();
  }
  renderSessionsList();
}

// ── Render a message bubble ───────────────────────────────
function appendMessage(role, content, animate = true, sources = null, idx = null) {
  const isAssistant = role === 'assistant';

  const row = document.createElement('div');
  row.className = `msg-row msg-row--${isAssistant ? 'assistant' : 'user'}`;
  if (!animate) row.style.animation = 'none';

  // Avatar (only for assistant/LLM)
  let avatarEl = null;
  if (isAssistant) {
    avatarEl = document.createElement('div');
    avatarEl.className = 'msg-avatar msg-avatar--assistant';
    avatarEl.innerHTML = `<img src="${BOT_AVATAR}" alt="INPPLC" />`;
  }

  // Bubble wrap
  const wrap = document.createElement('div');
  wrap.className = 'msg-bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = `msg-bubble msg-bubble--${isAssistant ? 'assistant' : 'user'}`;
  bubble.textContent = content;

  const meta = createMessageMeta(role, content, sources, idx);

  wrap.appendChild(bubble);
  wrap.appendChild(meta);

  if (avatarEl) {
    row.appendChild(avatarEl);
  }
  row.appendChild(wrap);

  messagesFeed.appendChild(row);
  return bubble; // Return bubble so streaming can update it
}

// ── Typing indicator ──────────────────────────────────────
function showTypingIndicator() {
  const row = document.createElement('div');
  row.className = 'msg-row msg-row--assistant';
  row.id = 'typingRow';

  const avatarEl = document.createElement('div');
  avatarEl.className = 'msg-avatar msg-avatar--assistant';
  avatarEl.innerHTML = `<img src="${BOT_AVATAR}" alt="INPPLC" />`;

  const wrap = document.createElement('div');
  wrap.className = 'msg-bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble msg-bubble--assistant';
  bubble.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;

  wrap.appendChild(bubble);
  row.appendChild(avatarEl);
  row.appendChild(wrap);
  messagesFeed.appendChild(row);
  scrollToBottom();
  return { row, bubble };
}

function removeTypingIndicator() {
  const row = document.getElementById('typingRow');
  if (row) row.remove();
}

// ── Send a message ────────────────────────────────────────
async function handleSend() {
  const text = messageInput.value.trim();
  if (!text || state.isStreaming) return;

  // Ensure session
  if (!state.currentSessionId) {
    await startNewChat();
  }

  // Show user bubble
  chatWelcome.classList.add('hidden');
  messagesFeed.style.display = 'flex';
  state.messages.push({ role: 'user', content: text });
  appendMessage('user', text, true, null, state.messages.length - 1);
  scrollToBottom();

  // Clear input
  messageInput.value = '';
  messageInput.style.height = 'auto';
  sendBtn.disabled = true;
  state.isStreaming = true;

  // Update sidebar preview
  const existing = state.sessions.find(s => s.session_id === state.currentSessionId);
  if (existing && existing.preview === 'New Chat') {
    existing.preview = text.substring(0, 48);
    renderSessionsList();
  }

  // Show typing indicator while waiting for first chunk
  const { row: typingRow, bubble: typingBubble } = showTypingIndicator();
  let firstChunk = true;
  let assistantBubble = null;
  let currentSources = null;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.currentSessionId,
        message: text,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const chunkStr = line.slice(6);
        if (chunkStr === '[DONE]') break;

        let chunk = chunkStr;
        let isSourceChunk = false;
        try {
          const parsed = JSON.parse(chunkStr);
          if (parsed && parsed.sources !== undefined) {
            currentSources = parsed.sources;
            isSourceChunk = true;
          } else if (parsed && parsed.token !== undefined) {
            chunk = parsed.token;
          }
        } catch(e) {}

        if (isSourceChunk) continue;

        if (firstChunk) {
          // Replace typing indicator with real bubble
          removeTypingIndicator();
          // Create streaming bubble directly
          const streamRow = document.createElement('div');
          streamRow.className = 'msg-row msg-row--assistant';
          const avatarEl = document.createElement('div');
          avatarEl.className = 'msg-avatar msg-avatar--assistant';
          avatarEl.innerHTML = `<img src="${BOT_AVATAR}" alt="INPPLC" />`;

          const wrap = document.createElement('div');
          wrap.className = 'msg-bubble-wrap';

          assistantBubble = document.createElement('div');
          assistantBubble.className = 'msg-bubble msg-bubble--assistant';

          // Temp simple meta row with copy button
          const meta = document.createElement('div');
          meta.className = 'msg-meta';
          const copyBtn = document.createElement('button');
          copyBtn.className = 'msg-action-btn';
          copyBtn.title = 'Copier';
          copyBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copier</span>
          `;
          copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(fullText).then(() => showToast('Copié !', 1600));
          });
          meta.appendChild(copyBtn);

          wrap.appendChild(assistantBubble);
          wrap.appendChild(meta);
          streamRow.appendChild(avatarEl);
          streamRow.appendChild(wrap);
          messagesFeed.appendChild(streamRow);
          firstChunk = false;
        }

        fullText += chunk;
        if (assistantBubble) assistantBubble.textContent = fullText;
        scrollToBottom();
      }
    }

    // Save assistant message to state.messages
    state.messages.push({ role: 'assistant', content: fullText, sources: currentSources });
    const assistantIdx = state.messages.length - 1;

    // Render final actions on assistant message
    if (assistantBubble) {
      const wrap = assistantBubble.closest('.msg-bubble-wrap');
      if (wrap) {
        const oldMeta = wrap.querySelector('.msg-meta');
        if (oldMeta) oldMeta.remove();
        const newMeta = createMessageMeta('assistant', fullText, currentSources, assistantIdx);
        wrap.appendChild(newMeta);
      }
    }

  } catch (err) {
    removeTypingIndicator();
    appendMessage('assistant', `⚠️ Une erreur est survenue lors de la connexion. Veuillez réessayer. / حدث خطأ أثناء الاتصال. الرجاء المحاولة مرة أخرى.`);
    scrollToBottom();
  } finally {
    state.isStreaming = false;
    sendBtn.disabled = messageInput.value.trim() === '';
  }
}

// ── Scroll helpers ────────────────────────────────────────
function scrollToBottom() {
  const chatArea = document.getElementById('chatArea');
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// ── Clear current chat ────────────────────────────────────
clearChatBtn.addEventListener('click', async () => {
  if (!state.currentSessionId) return;
  await deleteSession(state.currentSessionId);
  await startNewChat();
  showToast('Chat cleared');
});

// ── New chat button ───────────────────────────────────────
newChatBtn.addEventListener('click', startNewChat);

// ── Recent chats button ───────────────────────────────────
recentChatsBtn.addEventListener('click', async () => {
  await fetchSessions();
  showToast('Sessions refreshed');
});

// ── Send button ───────────────────────────────────────────
sendBtn.addEventListener('click', handleSend);

// ── Sidebar search ────────────────────────────────────────
sidebarSearch.addEventListener('input', (e) => {
  renderSessionsList(e.target.value);
});

// ── Topbar search (highlight messages) ───────────────────
topbarSearch.addEventListener('input', (e) => {
  const q = e.target.value.trim();
  // Reset all bubbles
  document.querySelectorAll('.msg-bubble').forEach(b => {
    if (b._rawText) b.textContent = b._rawText;
  });
  if (!q) return;
  document.querySelectorAll('.msg-bubble').forEach(b => {
    const text = b.textContent;
    if (!b._rawText) b._rawText = text;
    if (text.toLowerCase().includes(q.toLowerCase())) {
      b.innerHTML = highlightText(escapeHtml(text), q);
    }
  });
});

// ── Warning button ────────────────────────────────────────
warningBtn.addEventListener('click', () => {
  showToast('Please verify important information with official INPPLC sources.', 4000);
});

// ── Close Audit Panel ─────────────────────────────────────
if (auditPanelCloseBtn) {
  auditPanelCloseBtn.addEventListener('click', () => {
    if (auditTrailPanel) auditTrailPanel.classList.add('collapsed');
  });
}

// ── Floating search button (mobile) ──────────────────────
floatingSearchBtn.addEventListener('click', () => {
  openSidebar();
  setTimeout(() => sidebarSearch.focus(), 300);
});

// ── Responsive: close sidebar on overlay click (mobile) ──
document.addEventListener('click', (e) => {
  if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')) {
    if (!sidebar.contains(e.target) && e.target !== sidebarToggleBtn) {
      closeSidebar();
    }
  }
});

// ── Init ──────────────────────────────────────────────────
(async function init() {
  await fetchSessions();
  if (state.sessions.length > 0) {
    // Pre-load most recent session
    const latest = state.sessions[0];
    await switchToSession(latest.session_id, latest.preview);
  } else {
    await startNewChat();
  }
})();
