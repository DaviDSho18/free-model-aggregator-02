// free-model-aggregator-02: one server, two client wires, and the best free model behind every answer.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from the repository's root, with Node 22's standard
//   library alone (nothing is installed; TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; every model call goes through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the key pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries the free model list (each model's gateway id,
//   its published rating in milli-points, what it can do, its context window) and the six prompts.
//
// WHAT THIS ENTRY DOES WITH IT
//   1. Reads the brief, ranks the free models that meet each prompt's requirements by the brief's own
//      rating, and serves each prompt from the best one that answers.
//   2. Retries a refused prompt on the next-ranked model, and reports every refusal it recovered from in
//      the optional `x-makefun-upstream-failures` header, which is the only way the failover column can
//      see a recovered refusal.
//   3. Cools a model down after it refuses, counted in CALLS rather than in wall-clock time, so one
//      rate-limited model cannot take every later prompt down with it and a replay still reproduces.
//   4. Registers the exact fourteen dialable ids and answers GET /v1/models with that same set on both
//      auth headers.
//   5. Polls GET $MAKEFUN_ENV_URL/work and exits 0 the moment it says done.
//
// DETERMINISM
//   Every route is a function of the brief, the requests and the gateway's answers, never of the wall
//   clock and never of unseeded randomness: a live pass and its replay must leave the same state export,
//   byte for byte. Nothing here reads Date.now(), new Date() or Math.random().
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   lib/routing.ts     quality-first candidate ranking, per-model cooldowns, the refusal trail.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  CandidateTable,
  Cooldowns,
  planFor,
  refusalCode,
  refusalTrail,
  type Refusal,
} from './lib/routing.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most models one prompt may be tried on before it is given up on. */
const MAX_ATTEMPTS = 4;
/** The most output tokens this server asks for when the client named no cap: the gateway refuses none. */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Each requirement shape's candidates, ranked once from the brief. */
const table = new CandidateTable(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** The models sitting out their cooldown, counted in calls. */
const cooldowns = new Cooldowns();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return brief.models
    .filter((model) => model.free)
    .map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, for a brief that holds no prompt row for the request. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway call: the Chat Completions route, naming the model this entry really dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** What one prompt's serving attempt produced: the answer to relay, or the refusals that ended it. */
type Served =
  | { ok: true; model: string; upstream: Response; refusals: Refusal[] }
  | { ok: false; refusals: Refusal[]; status: number; body: string };

/**
 * Serve one prompt from the best free model that answers it. The candidates are the brief's free models
 * that meet the requirements, ranked by the brief's rating; a refusal moves on to the next one, and the
 * refusals are carried back so the answer can report the trail it recovered from.
 *
 * The body builder is called per candidate because the model id is part of the body the gateway sees.
 */
async function serve(requires: Requirement, build: (model: string) => Json): Promise<Served> {
  const candidates = planFor(table.candidates(requires), cooldowns, MAX_ATTEMPTS);
  const refusals: Refusal[] = [];
  let lastStatus = 502;
  let lastBody = '';

  for (const model of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(model.id, build(model.id));
    } catch (err) {
      // A transport failure is not a provider refusal: it is this server's own error, and it is not
      // reported in the trail (the column counts the gateway's refusals, not the network's).
      lastStatus = 502;
      lastBody = (err as Error).message;
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      cooldowns.advance();
      return { ok: true, model: model.id, upstream, refusals };
    }
    // A 4xx or a 5xx: the provider's own refusal, relayed to us as it sent it. Sit the model out and
    // try the next-ranked one.
    const text = await upstream.text();
    refusals.push({ status: upstream.status, code: refusalCode(upstream.status, text), model: model.id });
    cooldowns.penalise(model.id);
    lastStatus = upstream.status;
    lastBody = text;
  }

  cooldowns.advance();
  return { ok: false, refusals, status: lastStatus, body: lastBody };
}

/** The refusal trail as a response header, when there is one to report. */
function trailHeader(refusals: readonly Refusal[]): Record<string, string> {
  return refusals.length === 0 ? {} : { 'x-makefun-upstream-failures': refusalTrail(refusals) };
}

/** A prompt this server could not serve: the last refusal, in the client's own wire. */
function giveUp(res: ServerResponse, wire: 'openai' | 'anthropic', served: Extract<Served, { ok: false }>): void {
  const headers = trailHeader(served.refusals);
  const detail = served.body === '' ? '' : `: ${served.body.slice(0, 500)}`;
  if (wire === 'anthropic') {
    const body = anthropicError('api_error', `every free model that meets this prompt refused (last HTTP ${served.status})${detail}`);
    const text = JSON.stringify(body);
    res.writeHead(502, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
    res.end(text);
    return;
  }
  const body = openAIError(`every free model that meets this prompt refused (last HTTP ${served.status})${detail}`, 'upstream_error', null);
  const text = JSON.stringify(body);
  res.writeHead(502, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

/** A body with an output cap: the gateway refuses a call that names none. */
function withCap(body: Json): Json {
  if (typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number') return body;
  return { ...body, max_tokens: DEFAULT_MAX_TOKENS };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that answers. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (table.candidates(requires).length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  const served = await serve(requires, (model) => withCap({ ...body, model }));
  if (!served.ok) return giveUp(res, 'openai', served);
  const headers = trailHeader(served.refusals);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(served.upstream.body as ReadableStream<Uint8Array>, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await served.upstream.json();
  const text = JSON.stringify(openAIJsonWithModel(isRecord(answer) ? answer : {}, served.model));
  res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (table.candidates(requires).length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  const served = await serve(requires, (model) => withCap(anthropicToOpenAI(body, model)));
  if (!served.ok) return giveUp(res, 'anthropic', served);
  const headers = trailHeader(served.refusals);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(served.upstream.body as ReadableStream<Uint8Array>, served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await served.upstream.json();
  const text = JSON.stringify(openAIToAnthropic(isRecord(answer) ? answer : {}, served.model));
  res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
