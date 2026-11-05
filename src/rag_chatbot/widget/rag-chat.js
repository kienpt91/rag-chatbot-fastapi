/*
 * Drop-in chat widget: one classic script, no dependencies, no build step.
 * Configuration comes from the <script> tag's data-* attributes.
 * Safety rules: all text reaches the page through textContent only, and all
 * styling goes through the CSSOM, so a strict CSP (no 'unsafe-inline') works.
 */
(function () {
  'use strict';

  const script = document.currentScript || document.querySelector('script[data-api-url]');
  const opt = (name, fallback) => {
    const v = script ? script.getAttribute('data-' + name) : null;
    return typeof v === 'string' && v.trim() !== '' ? v.trim() : fallback;
  };

  // The API address must be an absolute http(s) URL, otherwise do nothing.
  let api = null;
  try {
    const u = new URL(opt('api-url', ''));
    if (u.protocol === 'http:' || u.protocol === 'https:') api = u.origin + u.pathname.replace(/\/+$/, '');
  } catch (e) {
    api = null;
  }
  if (!api) {
    console.warn('rag-chat: data-api-url must be an absolute http(s) URL; widget disabled.');
    return;
  }

  let token = opt('token', '');
  const tokenUrl = opt('token-url', '');
  const title = opt('title', 'Ask us');
  const greeting = opt('greeting', 'Hi! Ask me anything about this website.');
  const placeholder = opt('placeholder', 'Type your question');
  const sendLabel = opt('send-label', 'Send');
  const closeLabel = opt('close-label', 'Close');
  const openLabel = opt('open-label', 'Open chat');
  const errorText = opt('error-text', 'Sorry, something went wrong. Please try again.');
  const retryText = opt('retry-text', 'Please send your message again.');
  const sourcesLabel = opt('sources-label', 'Sources');
  // The accent is validated because it ends up in a CSS value.
  const accentRaw = opt('accent', '');
  const accent = /^#[0-9a-fA-F]{3,8}$/.test(accentRaw) || /^[a-zA-Z]{3,20}$/.test(accentRaw) ? accentRaw : '#1f6feb';
  const side = opt('position', 'right') === 'left' ? 'left' : 'right';
  const maxParsed = parseInt(opt('max-length', ''), 10);
  const maxLen = maxParsed > 0 ? Math.min(maxParsed, 4000) : 2000;

  const SESSION_KEY = 'rag-chat-session';
  const TIMEOUT_MS = 30000;
  const MAX_ANSWER = 8000;
  const MAX_CITATIONS = 5;
  const MAX_LINK_TEXT = 80;

  // The session id lives in memory too, so storage being blocked only costs persistence.
  let sessionId = '';
  try {
    const stored = window.sessionStorage.getItem(SESSION_KEY);
    if (typeof stored === 'string' && stored.length <= 200) sessionId = stored;
  } catch (e) {
    sessionId = '';
  }
  const setSession = (id) => {
    sessionId = id;
    try {
      if (id) window.sessionStorage.setItem(SESSION_KEY, id);
      else window.sessionStorage.removeItem(SESSION_KEY);
    } catch (e) {
      /* storage unavailable: keep the in-memory id only */
    }
  };

  // ---- DOM helpers -------------------------------------------------------
  const make = (tag, attrs, styles) => {
    const node = document.createElement(tag);
    Object.keys(attrs || {}).forEach((k) => node.setAttribute(k, attrs[k]));
    Object.keys(styles || {}).forEach((k) => node.style.setProperty(k, styles[k]));
    return node;
  };

  const launcher = make('button', { type: 'button', 'aria-label': openLabel, 'aria-expanded': 'false' }, {
    position: 'fixed', bottom: '16px', [side]: '16px', padding: '12px 18px', border: '0',
    'border-radius': '24px', 'background-color': accent, color: '#fff', cursor: 'pointer',
    'font-family': 'system-ui, sans-serif', 'font-size': '14px', 'z-index': '2147483000',
  });
  launcher.textContent = openLabel;

  const panel = make('div', { role: 'dialog', 'aria-label': title }, {
    display: 'none', position: 'fixed', bottom: '68px', [side]: '16px', width: '360px',
    'max-width': 'calc(100vw - 32px)', height: '480px', 'max-height': 'calc(100vh - 100px)',
    'flex-direction': 'column', background: '#fff', color: '#1b1f24', border: '1px solid #d0d7de',
    'border-radius': '12px', 'box-shadow': '0 8px 24px rgba(0,0,0,0.2)',
    'font-family': 'system-ui, sans-serif', 'font-size': '14px', 'z-index': '2147483000',
  });

  const header = make('div', {}, {
    display: 'flex', 'align-items': 'center', 'justify-content': 'space-between', padding: '10px 12px',
    'background-color': accent, color: '#fff', 'border-radius': '11px 11px 0 0', 'font-weight': '600',
  });
  const heading = make('span');
  heading.textContent = title;
  const closeBtn = make('button', { type: 'button', 'aria-label': closeLabel }, {
    border: '0', background: 'transparent', color: '#fff', cursor: 'pointer', 'font-size': '18px',
  });
  closeBtn.textContent = '\u00d7';
  header.append(heading, closeBtn);

  const log = make('div', { role: 'log', 'aria-live': 'polite' }, {
    flex: '1 1 auto', 'overflow-y': 'auto', padding: '10px', display: 'flex', 'flex-direction': 'column',
  });

  const footer = make('div', {}, { display: 'flex', gap: '8px', padding: '8px', 'border-top': '1px solid #d0d7de' });
  const input = make('textarea', { rows: '1', maxlength: String(maxLen), placeholder, 'aria-label': placeholder }, {
    flex: '1 1 auto', resize: 'none', padding: '8px', border: '1px solid #d0d7de', 'border-radius': '8px',
    font: 'inherit',
  });
  const sendBtn = make('button', { type: 'button' }, {
    padding: '8px 14px', border: '0', 'border-radius': '8px', 'background-color': accent, color: '#fff',
    cursor: 'pointer',
  });
  sendBtn.textContent = sendLabel;
  footer.append(input, sendBtn);
  panel.append(header, log, footer);

  // Every message is a wrapper holding one text node set via textContent.
  const addMessage = (kind, text) => {
    const styles = {
      margin: '0 0 8px', padding: '8px 10px', 'border-radius': '10px', 'max-width': '85%',
      'white-space': 'pre-wrap', 'overflow-wrap': 'anywhere', 'background-color': '#f1f3f5',
      'align-self': 'flex-start',
    };
    if (kind === 'user') Object.assign(styles, { 'background-color': accent, color: '#fff', 'align-self': 'flex-end' });
    if (kind === 'error') Object.assign(styles, { 'background-color': '#fef3f2', color: '#b42318' });
    const wrapper = make('div', {}, styles);
    wrapper.className = 'rag-chat-msg rag-chat-' + kind;
    const body = make('div');
    body.textContent = text;
    wrapper.appendChild(body);
    log.appendChild(wrapper);
    log.scrollTop = log.scrollHeight;
    return wrapper;
  };

  // Only http(s) URLs become links; anything else is shown as plain text.
  const addCitations = (message, citations) => {
    const list = make('ul', {}, { margin: '6px 0 0', padding: '0 0 0 18px', 'font-size': '12px' });
    for (let i = 0; i < Math.min(citations.length, MAX_CITATIONS); i++) {
      const c = citations[i];
      if (!c || typeof c !== 'object') continue;
      let url = null;
      try {
        const parsed = new URL(String(c.url));
        if (parsed.protocol === 'http:' || parsed.protocol === 'https:') url = parsed;
      } catch (e) {
        url = null;
      }
      const name = typeof c.title === 'string' ? c.title.trim() : '';
      const label = (name || (url ? url.hostname : '')).slice(0, MAX_LINK_TEXT);
      if (!label) continue;
      const item = make('li');
      const node = url
        ? make('a', { href: url.href, target: '_blank', rel: 'noopener noreferrer nofollow' }, { color: accent })
        : make('span');
      node.textContent = label;
      item.appendChild(node);
      list.appendChild(item);
    }
    if (list.children.length === 0) return;
    const caption = make('div', {}, { 'margin-top': '6px', 'font-size': '12px', 'font-weight': '600' });
    caption.textContent = sourcesLabel;
    message.append(caption, list);
  };

  // ---- Network -----------------------------------------------------------
  // One fetch with a hard timeout; the body is read inside the same window.
  const request = async (url, init) => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(url, Object.assign(
        { credentials: 'omit', mode: 'cors', referrerPolicy: 'no-referrer', signal: ctl.signal }, init));
      let data = null;
      try {
        data = await res.json();
      } catch (e) {
        data = null;
      }
      return { status: res.status, data };
    } finally {
      clearTimeout(timer);
    }
  };

  const getToken = async (force) => {
    if (token && !force) return token;
    if (!tokenUrl) throw new Error('no token source');
    const r = await request(tokenUrl, { method: 'GET' });
    if (r.status !== 200 || !r.data || typeof r.data.token !== 'string' || !r.data.token) {
      throw new Error('bad token response');
    }
    token = r.data.token;
    return token;
  };

  const post = async (body) => request(api + '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + (await getToken(false)) },
    body,
  });

  const errorCode = (d) => (d && typeof d === 'object' && d.error && typeof d.error === 'object' ? d.error.code : '');

  const handle = (r) => {
    const d = r.data;
    if (r.status === 200 && d && typeof d === 'object' && typeof d.answer === 'string') {
      if (typeof d.session_id === 'string' && d.session_id !== '' && d.session_id.length <= 200) {
        setSession(d.session_id);
      }
      input.value = '';
      const message = addMessage('assistant', d.answer.slice(0, MAX_ANSWER));
      if (d.grounded === true && Array.isArray(d.citations)) addCitations(message, d.citations);
      return;
    }
    const m = r.status !== 200 && d && d.error ? d.error.message : null;
    let text = typeof m === 'string' && m !== '' && m.length <= 300 ? m : errorText;
    if (r.status === 404 && errorCode(d) === 'session_not_found') {
      setSession('');
      text += ' ' + retryText;
    }
    addMessage('error', text);
  };

  // ---- Behaviour ---------------------------------------------------------
  let pending = false;
  let isOpen = false;

  const send = async () => {
    if (pending) return;
    const text = input.value.trim();
    if (!text) return;
    if (text.length > maxLen) {
      addMessage('error', errorText);
      return;
    }
    addMessage('user', text);
    pending = true;
    input.disabled = true;
    sendBtn.disabled = true;
    const typing = addMessage('typing', '...');
    try {
      const body = JSON.stringify(sessionId ? { message: text, session_id: sessionId } : { message: text });
      let r = await post(body);
      // An expired token is refreshed once, and the message is retried once.
      if (r.status === 401 && errorCode(r.data) === 'unauthorized' && tokenUrl) {
        await getToken(true);
        r = await post(body);
      }
      handle(r);
    } catch (e) {
      addMessage('error', errorText);
    } finally {
      typing.remove();
      pending = false;
      input.disabled = false;
      sendBtn.disabled = false;
      if (isOpen) input.focus();
    }
  };

  const setOpen = (open) => {
    isOpen = open;
    panel.style.setProperty('display', open ? 'flex' : 'none');
    launcher.setAttribute('aria-expanded', open ? 'true' : 'false');
    (open ? input : launcher).focus();
  };

  launcher.addEventListener('click', () => setOpen(!isOpen));
  closeBtn.addEventListener('click', () => setOpen(false));
  sendBtn.addEventListener('click', send);
  panel.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setOpen(false);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });

  const mount = () => {
    addMessage('assistant', greeting);
    document.body.appendChild(panel);
    document.body.appendChild(launcher);
  };
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);

  if (!window.RagChat) {
    window.RagChat = {
      open: () => setOpen(true),
      close: () => setOpen(false),
      toggle: () => setOpen(!isOpen),
    };
  }
})();
