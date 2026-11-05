// A minimal fake DOM, just big enough for the widget. Unsafe APIs fail loudly.

class TextNode {
  constructor(data) {
    this.nodeType = 3;
    this.data = String(data);
    this.parentNode = null;
  }

  get textContent() {
    return this.data;
  }
}

function forbid(name) {
  return {
    get() {
      throw new Error(`forbidden DOM API read: ${name}`);
    },
    set() {
      throw new Error(`forbidden DOM API write: ${name}`);
    },
  };
}

class Element {
  constructor(tagName, ownerEnv) {
    this.nodeType = 1;
    this.tagName = String(tagName).toLowerCase();
    this.parentNode = null;
    this.childNodes = [];
    this.attrs = new Map();
    this.listeners = new Map();
    this.style = {
      props: {},
      setProperty(name, value) {
        this.props[name] = String(value);
      },
    };
    this.className = '';
    this.value = '';
    this.disabled = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.env = ownerEnv;
  }

  get children() {
    return this.childNodes.filter((n) => n.nodeType === 1);
  }

  get textContent() {
    return this.childNodes.map((n) => n.textContent).join('');
  }

  set textContent(value) {
    for (const n of this.childNodes) n.parentNode = null;
    this.childNodes = [];
    this.appendChild(new TextNode(value));
  }

  appendChild(child) {
    if (child.parentNode) child.parentNode.childNodes = child.parentNode.childNodes.filter((n) => n !== child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  append(...nodes) {
    for (const n of nodes) this.appendChild(typeof n === 'string' ? new TextNode(n) : n);
  }

  remove() {
    if (!this.parentNode) return;
    this.parentNode.childNodes = this.parentNode.childNodes.filter((n) => n !== this);
    this.parentNode = null;
  }

  setAttribute(name, value) {
    const key = String(name).toLowerCase();
    if (key === 'style' || key.startsWith('on')) throw new Error(`forbidden attribute: ${name}`);
    this.attrs.set(key, String(value));
  }

  getAttribute(name) {
    const key = String(name).toLowerCase();
    return this.attrs.has(key) ? this.attrs.get(key) : null;
  }

  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }

  // Events bubble from the target up through the parents.
  dispatchEvent(event) {
    event.target = event.target || this;
    if (!event.preventDefault) {
      event.preventDefault = () => {
        event.defaultPrevented = true;
      };
    }
    for (let node = this; node; node = node.parentNode) {
      for (const fn of node.listeners.get(event.type) || []) fn(event);
    }
    return !event.defaultPrevented;
  }

  click() {
    if (!this.disabled) this.dispatchEvent({ type: 'click' });
  }

  focus() {
    if (!this.disabled) this.env.document.activeElement = this;
  }
}

for (const name of ['innerHTML', 'outerHTML', 'insertAdjacentHTML']) {
  Object.defineProperty(Element.prototype, name, forbid(name));
}

export function inMemoryStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => void data.set(k, String(v)),
    removeItem: (k) => void data.delete(k),
  };
}

export function throwingStorage() {
  const fail = () => {
    throw new Error('storage unavailable');
  };
  return { getItem: fail, setItem: fail, removeItem: fail };
}

// Depth-first walk over every node below (and including) root.
export function walk(root, visit) {
  visit(root);
  for (const child of root.childNodes || []) walk(child, visit);
}

export function createEnvironment({ scriptAttrs, fetch, storage, viaQuery = false } = {}) {
  const env = {};
  const timers = new Map();
  let nextTimer = 1;

  const script = scriptAttrs ? new Element('script', env) : null;
  if (script) for (const [k, v] of Object.entries(scriptAttrs)) script.setAttribute(k, v);

  const document = {
    body: new Element('body', env),
    currentScript: viaQuery ? null : script,
    activeElement: null,
    createElement: (tag) => new Element(tag, env),
    createTextNode: (text) => new TextNode(text),
    querySelector: (selector) => (script && selector.startsWith('script[data-api-url]') && script.getAttribute('data-api-url') !== null ? script : null),
    addEventListener() {},
    write() {
      throw new Error('forbidden DOM API: document.write');
    },
    writeln() {
      throw new Error('forbidden DOM API: document.writeln');
    },
  };

  const calls = [];
  const consoleSpy = { calls };
  for (const level of ['log', 'info', 'warn', 'error', 'debug']) {
    consoleSpy[level] = (...args) => void calls.push({ level, args });
  }

  Object.assign(env, {
    document,
    fetch,
    console: consoleSpy,
    URL,
    AbortController,
    JSON,
    sessionStorage: storage || inMemoryStorage(),
    // Timers never fire on their own; tests call fireTimers().
    timers,
    setTimeout(fn, delay) {
      const id = nextTimer++;
      timers.set(id, { fn, delay });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    fireTimers() {
      for (const [id, t] of [...timers]) {
        timers.delete(id);
        t.fn();
      }
    },
  });
  env.window = env;
  return env;
}
