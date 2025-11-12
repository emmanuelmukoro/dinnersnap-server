// api/analyze.js
// ESM-only. Works with "type":"module" in package.json".
// Vision -> cleaned pantry -> Spoonacular + LLM (in parallel) -> emergency fallback.

export const config = {
  runtime: "nodejs",
  maxDuration: 25,
  memory: 1024,
  regions: ["lhr1"],
};

/* ----------------------- Env ----------------------- */
const GCV_KEY = process.env.GCV_KEY;
const SPOON_KEY = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

/* ----------------------- Helpers ----------------------- */
const json = (status, obj) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });

const nowMs = () => Number(process.hrtime.bigint() / 1000000n);

function withTimeout(run, ms, label = "timeout") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return Promise.race([
    run(controller.signal),
    new Promise((_, rej) => setTimeout(() => rej(new Error(label)), ms + 10)),
  ]).finally(() => clearTimeout(timer));
}

function tfetch(url, opts = {}, ms = 2500, label = "fetch-timeout") {
  return withTimeout((signal) => fetch(url, { ...opts, signal }), ms, label);
}

/* ----------------------- Classifiers ----------------------- */
const DESSERT = new Set(["dessert","pudding","ice cream","smoothie","shake","cookie","brownie","cupcake","cake","muffin","pancake","waffle","jam","jelly","compote","truffle","fudge","sorbet","parfait","custard","tart","shortbread","licorice","cobbler"]);
const DRINK   = new Set(["drink","beverage","mocktail","cocktail","margarita","mojito","spritzer","soda","punch","toddy"]);
const ALCOHOL = new Set(["brandy","rum","vodka","gin","whisky","whiskey","bourbon","tequila","wine","liqueur","amaretto","cognac","port","sherry"]);
const GENERIC = new Set(["pasta","rice","bread","flour","sugar","oil","salt","pepper"]);
const MEAT    = new Set(["chicken","beef","pork","lamb","bacon","ham","turkey","shrimp","prawn","salmon","tuna","fish"]);
const PASTA   = new Set(["pasta","spaghetti","macaroni","penne","farfalle","orzo","fusilli","linguine","tagliatelle"]);

const titleHasAny = (title, set) => {
  const t = String(title || "").toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
};
const hasSpecificTermInTitleOrIngredients = (item, specificTerms) => {
  const t = String(item.title || "").toLowerCase();
  const ings = (item.extendedIngredients || []).map((i) => String(i.name || "").toLowerCase());
  return specificTerms.some((p) => t.includes(p) || ings.some((n) => n.includes(p)));
};

/* ----------------------- Vision (2.5s) ----------------------- */
async function callVision(imageBase64) {
  if (!GCV_KEY)
    return { ocrTokens: [], labels: [], objects: [], error: "no GCV_KEY" };

  const body = {
    requests: [
      {
        image: { content: imageBase64.replace(/^data:image\/\w+;base64,/, "") },
        features: [
          { type: "TEXT_DETECTION", maxResults: 1 },
          { type: "LABEL_DETECTION", maxResults: 10 },
          { type: "OBJECT_LOCALIZATION", maxResults: 10 },
        ],
      },
    ],
  };

  const r = await tfetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    },
    2500,
    "vision-timeout"
  );

  const j = await r.json();
  const res = j?.responses?.[0] || {};

  const rawText = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = rawText.split(/[^a-z]+/g).map((t) => t.trim()).filter(Boolean);

  const FOODISH = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen","beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash","chickpea","chickpeas","feta","rice","pasta","spaghetti","oats","milk","yogurt","cheese","lemon","orange","pepper","spinach","mushroom","potato","avocado","coconut","cream",
  ]);

  const ocrTokens = RAW_TOKENS.filter((t) => t.length >= 4 && FOODISH.has(t));
  const labels  = (res.labelAnnotations || []).map((x) => x.description?.toLowerCase()).filter(Boolean);
  const objects = (res.localizedObjectAnnotations || []).map((x) => x.name?.toLowerCase()).filter(Boolean);
  return { ocrTokens, labels, objects };
}

/* ----------------------- Pantry cleanup ----------------------- */
const WHITELIST = [
  "banana","broccoli","chickpeas","beans","kidney beans","tomatoes","onion","garlic","ginger","olive oil","pasta","spaghetti","courgette","feta","orzo","rice","egg","spring onion","lemon","orange","avocado","coconut milk","coconut cream","butternut squash","squash","carrot","pepper","bell pepper","spinach","potato","mushroom","cheese","cheddar","yogurt","milk","almond milk","chicken","chicken breast","beef","pork","fish","salmon","tuna","bread","tortilla","wrap","lentils","cucumber","lettuce","cabbage","kale","apple","pear","oats","flour","sugar","butter","salt","pepper","stock cube","curry powder","mixed dried herbs","garam masala","maggi seasoning","jeera","cumin","cloves",
];
const MAP = {
  bananas: "banana",
  chickpea: "chickpeas",
  garbanzo: "chickpeas",
  "garbanzo bean": "chickpeas",
  "garbanzo beans": "chickpeas",
  zucchini: "courgette",
  courgettes: "courgette",
  tomato: "tomatoes",
  onions: "onion",
  eggs: "egg",
  brockley: "broccoli",
  maggi: "maggi seasoning",
  "jeera powder": "jeera",
};

const uniqLower = (arr) => [...new Set(arr.map((x) => String(x).toLowerCase()))];
function lev(a,b){const m=[];for(let i=0;i<=b.length;i++)m[i]=[i];for(let j=0;j<=a.length;j++)m[0][j]=j;for(let i=1;i<=b.length;i++){for(let j=1;j<=a.length;j++){m[i][j]=Math.min(m[i-1][j]+1,m[i][j-1]+1,m[i-1][j-1]+(b[i-1]===a[j-1]?0:1));}}return m[b.length][a.length];}
function nearestWhitelistStrict(term){let best=null,bestDist=2;for(const w of WHITELIST){const d=lev(term,w);if(d<bestDist){bestDist=d;best=w;}}return bestDist<=1?best:null;}
function cleanPantry(raw){const out=new Set();for(const t0 of uniqLower(raw)){const t=MAP[t0]||t0;if(WHITELIST.includes(t)){out.add(t);continue;}if(t.length>=5){const near=nearestWhitelistStrict(t);if(near){out.add(near);continue;}}}return [...out];}

/* ----------------------- Spoonacular ----------------------- */
async function spoonacularRecipes(pantry, prefs) {
  if (!SPOON_KEY) return { results: [], info: { note: "no SPOON_KEY" } };

  const hasMeat = pantry.some((p) => MEAT.has(p));
  const include = pantry.join(",");
  const timeCap = Math.max(10, (prefs?.time ?? 25) + 10);

  const url = new URL("https://api.spoonacular.com/recipes/complexSearch");
  url.searchParams.set("apiKey", SPOON_KEY);
  url.searchParams.set("includeIngredients", include);
  url.searchParams.set("instructionsRequired", "true");
  url.searchParams.set("addRecipeInformation", "true");
  url.searchParams.set("sort", "max-used-ingredients");
  url.searchParams.set("number", "18");
  url.searchParams.set("ignorePantry", "true");
  url.searchParams.set("type", "main course");
  url.searchParams.set("excludeIngredients", Array.from(ALCOHOL).join(","));
  url.searchParams.set("maxReadyTime", String(timeCap));
  if (!hasMeat) url.searchParams.set("diet", "vegetarian");

  let j = {};
  try {
    const r = await tfetch(url.toString(), {}, 2000, "spoon-timeout");
    j = await r.json();
  } catch (e) {
    return { results: [], info: { error: String(e?.message || e) } };
  }

  const raw = Array.isArray(j?.results) ? j.results : [];
  const specific = pantry.filter((p) => !GENERIC.has(p));

  const filtered = raw.filter((it) => {
    const title = String(it.title || "").toLowerCase();
    const dish = (it.dishTypes || []).map((d) => String(d).toLowerCase());
    if (titleHasAny(title, DESSERT)) return false;
    if (titleHasAny(title, DRINK)) return false;
    if (titleHasAny(title, ALCOHOL)) return false;
    if (dish.some((d) => ["drink","beverage","cocktail","dessert"].includes(d))) return false;
    const used = it.usedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 999;
    if (used < 2) return false;
    if (time > timeCap) return false;
    if (specific.length >= 2 && !hasSpecificTermInTitleOrIngredients(it, specific)) return false;
    if (/shrimp|prawn|salmon|tuna|anchovy|sardine/.test(title) &&
        !pantry.some((p) => ["shrimp","prawn","salmon","tuna","fish"].includes(p))) return false;
    return true;
  });

  const scored = filtered.map((it) => {
    const used = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 30;
    const score =
      0.7 * (used / (used + missed + 1)) +
      0.3 * (1 - Math.min(time, 60) / 60);
    return {
      id: String(it.id),
      title: it.title,
      time,
      energy: "hob",
      cost: 2.5,
      score: Math.round(score * 100) / 100,
      ingredients: (it.extendedIngredients || []).map((ing) => {
        const nm = String(ing.name || "").toLowerCase();
        return { name: nm, have: pantry.includes(nm) };
      }),
      steps: (it.analyzedInstructions?.[0]?.steps || []).map((s, i) => ({
        id: `${it.id}-s${i}`,
        text: s.step,
      })),
      badges: ["web"],
    };
  });
  return { results: scored, info: { spoonacularRawCount: raw.length, kept: scored.length, timeCap } };
}

/* ----------------------- LLM ----------------------- */
async function llmRecipe(pantry, prefs) {
  if (!OPENAI_API_KEY)
    return { recipe: null, info: { reason: "no OPENAI_API_KEY" } };

  const body = {
    model: "gpt-4o-mini",
    temperature: 0.45,
    max_tokens: 600,
    messages: [
      { role: "system", content: "You write concise, savoury DINNER recipes only." },
      {
        role: "user",
        content: `Create ONE savoury DINNER recipe using ${pantry.join(", ")}.`,
      },
    ],
  };

  try {
    const r = await tfetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify(body),
      },
      2500,
      "llm-timeout"
    );
    const j = await r.json();
    const txt = j?.choices?.[0]?.message?.content || "";
    const m = txt.match(/\{[\s\S]*\}$/);
    if (!m) return { recipe: null, info: { error: "no JSON in completion" } };
    const parsed = JSON.parse(m[0]);
    parsed.id = parsed.id || `llm-${Date.now()}`;
    parsed.energy = parsed.energy || "hob";
    parsed.cost = parsed.cost ?? 2.0;
    parsed.badges = Array.isArray(parsed.badges) ? parsed.badges : ["llm"];
    parsed.ingredients = (parsed.ingredients || []).map((i) => {
      const nm = String(i.name || "").toLowerCase();
      return { name: nm, have: pantry.includes(nm) };
    });
    parsed.steps = (parsed.steps || []).map((s, i) => ({
      id: s.id || `step-${i}`,
      text: s.text || String(s),
    }));
    return { recipe: parsed, info: { ok: true } };
  } catch (e) {
    return { recipe: null, info: { error: String(e?.message || e) } };
  }
}

/* ----------------------- Emergency ----------------------- */
function emergencyRecipe(pantry) {
  const titleBits = [];
  if (pantry.some((p) => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (pantry.includes("coconut cream")) titleBits.push("Coconut");
  if (pantry.some((p) => PASTA.has(p))) titleBits.push("Spaghetti");
  const title =
    (titleBits.length ? titleBits.join(" ") : "Pantry") + " Savoury Skillet";
  const baseIngs = [
    { name: "onion (or onion powder)", have: pantry.includes("onion") },
    { name: "garlic (or garlic powder)", have: pantry.includes("garlic") },
    { name: "ginger (or ground ginger)", have: pantry.includes("ginger") },
    { name: "mixed dried herbs", have: pantry.includes("mixed dried herbs") },
    { name: "stock cube", have: pantry.includes("stock cube") },
    { name: "lemon or vinegar", have: pantry.includes("lemon") || pantry.includes("vinegar") },
    { name: "salt & black pepper", have: pantry.includes("salt") || pantry.includes("black pepper") },
  ];
  const uniqPantry = pantry.map((p) => ({ name: p, have: true }));
  return {
    id: `local-${Date.now()}`,
    title,
    time: 15,
    cost: 2.0,
    energy: "hob",
    ingredients: [...uniqPantry, ...baseIngs],
    steps: [
      { id: "s1", text: "Heat oil; add onion, garlic & ginger. Cook 2–3 min." },
      { id: "s2", text: "Add pantry items and simmer 6–8 min." },
    ],
    badges: ["local"],
  };
}

/* ----------------------- Handler ----------------------- */
export default async function handler(req) {
  const t0 = nowMs();
  if (req.method !== "POST") return json(405, { error: "POST only" });

  let body;
  try {
    body = await req.json();
  } catch {
    return json(400, { error: "invalid JSON" });
  }

  const { imageBase64, pantryOverride, prefs = {} } = body || {};
  const watchdogMs = 12000;

  const main = async () => {
    const tStart = nowMs();
    let pantry = [];
    let source = "";
    let pantryFrom = {};

    if (Array.isArray(pantryOverride) && pantryOverride.length) {
      pantry = cleanPantry(pantryOverride);
      source = "pantryOverride";
    } else if (imageBase64) {
      try {
        const v0 = nowMs();
        const visRes = await withTimeout(() => callVision(imageBase64), 2500, "vision-timeout");
        pantryFrom = { ocr: visRes.ocrTokens, labels: visRes.labels, objects: visRes.objects };
        pantry = cleanPantry([...(visRes.ocrTokens||[]),...(visRes.labels||[]),...(visRes.objects||[])]);
        source = "vision";
        console.log(`vision ok in ${nowMs() - v0}ms pantry=`, pantry);
      } catch (e) {
        source = "vision-failed";
        pantryFrom = { error: String(e?.message || e) };
        pantry = [];
      }
    } else {
      return json(400, { error: "imageBase64 required (or pantryOverride
