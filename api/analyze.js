// api/analyze.js
// Vision → cleaned pantry → Spoonacular (dinner-only, filtered) → LLM fallback (seasoned) → Emergency local

// --- Vercel runtime config ---
export const config = {
  runtime: "nodejs",   // IMPORTANT: nodejs (not edge)
  maxDuration: 25,
  memory: 1024,
  regions: ["lhr1"],   // London (optional; remove if you prefer auto)
};

// --- Env keys ---
const GCV_KEY = process.env.GCV_KEY;
const SPOON_KEY = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// --- Small helpers ---
const json = (status, obj) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });

// ---- timeouts (hard caps so the function never hits Vercel's overall limit) ----
function withTimeout(promiseFn, ms, label = "timeout") {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  return Promise.race([
    promiseFn(ctrl.signal),
    new Promise((_, rej) => setTimeout(() => rej(new Error(label)), ms + 5)),
  ]).finally(() => clearTimeout(t));
}

async function tfetch(url, opts = {}, ms = 3000, label = `fetch timeout ${ms}ms`) {
  return withTimeout(
    (signal) => fetch(url, { ...opts, signal }),
    ms,
    label
  );
}

// ---------- classifiers ----------
const DESSERT_WORDS = new Set([
  "dessert","pudding","ice cream","smoothie","shake","cookie","brownie","cupcake",
  "cake","muffin","pancake","waffle","jam","jelly","compote","truffle","fudge",
  "sorbet","parfait","custard","tart","shortbread","licorice","cobbler"
]);
const DRINK_WORDS = new Set(["drink","cocktail","beverage","mocktail","soda","punch","spritzer","toddy","cobbler"]);
const ALCOHOL_WORDS = new Set(["brandy","rum","vodka","gin","whisky","whiskey","bourbon","tequila","wine","liqueur","amaretto","cognac","port","sherry"]);
const GENERIC_WORDS = new Set(["pasta","rice","bread","flour","sugar","oil","salt","pepper"]);
const MEAT_WORDS = new Set(["chicken","beef","pork","lamb","bacon","ham","turkey","shrimp","prawn","salmon","tuna","fish"]);
const PASTA_WORDS = new Set(["pasta","spaghetti","macaroni","penne","farfalle","orzo","fusilli","linguine","tagliatelle"]);

function titleHasAny(title, set) {
  const t = String(title || "").toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
}
function hasSpecificTermInTitleOrIngredients(item, specificTerms) {
  const t = String(item.title || "").toLowerCase();
  const ings = (item.extendedIngredients || []).map(i => String(i.name || "").toLowerCase());
  return specificTerms.some(p => t.includes(p) || ings.some(n => n.includes(p)));
}

// ---------- Vision (with OCR/labels/objects and time cap) ----------
async function callVision(imageBase64) {
  const body = {
    requests: [{
      image: { content: imageBase64.replace(/^data:image\/\w+;base64,/, "") },
      features: [
        { type: "TEXT_DETECTION", maxResults: 1 },
        { type: "LABEL_DETECTION", maxResults: 10 },
        { type: "OBJECT_LOCALIZATION", maxResults: 10 },
      ],
    }],
  };

  const r = await tfetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    },
    3500, // <-- A) Vision capped at 3.5s
    "vision timeout"
  );
  const j = await r.json();
  const res = j?.responses?.[0] || {};

  // OCR tokens (aggressive cleanup)
  const ocrRaw = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = ocrRaw
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter((t) => t.length >= 4); // ignore tiny tokens

  const FOODISH_HINTS = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","spaghetti","oats","milk","yogurt","cheese",
    "lemon","orange","pepper","spinach","mushroom","potato","avocado","coconut","cream"
  ]);

  // keep token if it looks food-ish
  const ocrTokens = RAW_TOKENS.filter(t => FOODISH_HINTS.has(t));

  const labels  = (res.labelAnnotations || []).map(x => x.description?.toLowerCase()).filter(Boolean);
  const objects = (res.localizedObjectAnnotations || []).map(x => x.name?.toLowerCase()).filter(Boolean);

  return { ocrTokens, labels, objects };
}

// ---------- pantry cleanup ----------
const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta","spaghetti",
  "courgette","feta","orzo","rice","egg","spring onion","lemon","orange","avocado","coconut cream",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk",
  "chicken","chicken breast","beef","pork","fish","salmon","tuna",
  "bread","tortilla","wrap","beans","kidney beans","black beans","lentils",
  "cucumber","lettuce","cabbage","kale","apple","pear","oats","flour","sugar","butter","salt","pepper"
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
  brockley: "broccoli", // observed typo
};
function uniqLower(a){const s=new Set();for(const x of a)s.add(String(x).toLowerCase());return [...s];}
function lev(a,b){const m=[];for(let i=0;i<=b.length;i++)m[i]=[i];for(let j=0;j<=a.length;j++)m[0][j]=j;for(let i=1;i<=b.length;i++){for(let j=1;j<=a.length;j++){m[i][j]=Math.min(m[i-1][j]+1,m[i][j-1]+1,m[i-1][j-1]+(b.charAt(i-1)===a.charAt(j-1)?0:1));}}return m[b.length][a.length];}
function nearestWhitelistStrict(term){let best=null,bestDist=2;for(const w of WHITELIST){const d=lev(term,w);if(d<bestDist){bestDist=d;best=w;}}return bestDist<=1?best:null;}
function cleanPantry(rawTerms){
  const out=new Set();
  for(const tRaw of uniqLower(rawTerms)){
    const t = MAP[tRaw] || tRaw;
    if(WHITELIST.includes(t)){ out.add(t); continue; }
    if(t.length>=5){
      const nearest=nearestWhitelistStrict(t);
      if(nearest){ out.add(nearest); continue; }
    }
  }
  return [...out];
}

// ---------- Spoonacular (dinner-mode, 2.5s cap, filter nonsense) ----------
async function spoonacularRecipes(pantry, prefs){
  if(!SPOON_KEY) return { results: [], extra:{ note:"no SPOON_KEY" } };

  const hasMeat = pantry.some(p => MEAT_WORDS.has(p));
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
  url.searchParams.set("type", "main course");                 // dinner only
  url.searchParams.set("excludeIngredients", Array.from(ALCOHOL_WORDS).join(",")); // no alcohol
  url.searchParams.set("maxReadyTime", String(timeCap));
  if(!hasMeat) url.searchParams.set("diet", "vegetarian");

  let j = {};
  try {
    const r = await tfetch(url.toString(), {}, 2500, "spoonacular timeout"); // <-- B) 2.5s cap
    j = await r.json();
  } catch(e) {
    return { results: [], extra: { error: String(e?.message||e) } };
  }

  const raw = Array.isArray(j?.results) ? j.results : [];
  const pantrySpecific = pantry.filter(p => !GENERIC_WORDS.has(p));

  const filtered = raw.filter(it => {
    const title = String(it.title||"").toLowerCase();
    const dishTypes = (it.dishTypes || []).map(d => String(d).toLowerCase());

    if (titleHasAny(title,DESSERT_WORDS)) return false;
    if (titleHasAny(title,DRINK_WORDS))   return false;
    if (titleHasAny(title,ALCOHOL_WORDS)) return false;
    if (dishTypes.some(d => ["drink","beverage","cocktail","dessert"].includes(d))) return false;

    const used   = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time   = it.readyInMinutes ?? 999;
    if (used < 2) return false;
    if (time > timeCap) return false;

    if (pantrySpecific.length > 0 && !hasSpecificTermInTitleOrIngredients(it, pantrySpecific)) return false;

    if (title.includes("macaroni and cheese")){
      const hasDairy = pantry.some(p => ["cheese","milk","cream","yogurt"].includes(p));
      if (!hasDairy) return false;
    }

    if ((/shrimp|prawn|salmon|tuna|anchovy|sardine/).test(title) &&
        !pantry.some(p => ["shrimp","prawn","salmon","tuna","fish"].includes(p)))
      return false;

    return true;
  });

  const scored = filtered.map((it) => {
    const used   = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time   = it.readyInMinutes ?? 30;
    const score  = 0.7 * (used / (used + missed + 1)) + 0.3 * (1 - Math.min(time,60)/60);
    return {
      id: String(it.id),
      title: it.title,
      time,
      energy: "hob",
      cost: 2.5,
      score: Math.round(score*100)/100,
      ingredients: (it.extendedIngredients || []).map(ing => {
        const nm = String(ing.name||"").toLowerCase();
        return { name: nm, have: pantry.includes(nm) };
      }),
      steps: ((it.analyzedInstructions?.[0]?.steps)||[]).map((s,i)=>({ id:`${it.id}-s${i}`, text:s.step })),
      badges: ["web"],
    };
  });

  return { results: scored, extra: { spoonacularRawCount: raw.length, kept: scored.length, timeCap } };
}

// ---------- LLM fallback (savory + seasoned, 3s cap) ----------
function buildPrompt(pantry, prefs) {
  const core = pantry.join(", ");
  const minutes  = Math.min(Math.max(prefs?.time || 25, 10), 40);
  const servings = Math.min(Math.max(prefs?.servings || 2, 1), 6);
  const diet     = prefs?.diet || "none";
  const energy   = prefs?.energyMode || "hob";

  return `
Create ONE savoury DINNER recipe (no drinks, no desserts, no sweets) using primarily: ${core}.
Assume common staples available: salt, pepper, oil, stock cube/paste, onion, garlic, chilli, ginger, smoked paprika,
curry powder, dried herbs, soy sauce, lemon/lime, vinegar.
Ensure flavour: include aromatics (onion/garlic/ginger), an acid or freshness (lemon/lime/soy), and at least 3 seasonings from the staples.
Constraints:
- Ready in <= ${minutes} minutes
- Servings: ${servings}
- Diet: ${diet}
- Cooking method prefers: ${energy}

Return JSON with:
{
  "id": "llm-<some-id>",
  "title": "...",
  "time": ${minutes},
  "cost": 3,
  "energy": "${energy}",
  "ingredients": [{"name":"...", "have": true|false}, ...],
  "steps": [{"id":"s1","text":"..."}, ...],
  "badges": ["under-30","budget"]
}
Only return JSON.`;
}

async function llmRecipe(pantry, prefs){
  if(!OPENAI_API_KEY) return { recipe:null, reasonNoLLM:"OPENAI_API_KEY missing" };

  const body = {
    model: "gpt-4o-mini",
    temperature: 0.45,
    max_tokens: 600,
    messages: [
      { role: "system", content: "You are a skilled home cook who writes concise, savoury DINNER recipes only." },
      { role: "user", content: buildPrompt(pantry, prefs) }
    ]
  };

  try{
    const r = await tfetch(
      "https://api.openai.com/v1/chat/completions",
      {
        method:"POST",
        headers:{ "content-type":"application/json", authorization:`Bearer ${OPENAI_API_KEY}` },
        body: JSON.stringify(body)
      },
      3000, // <-- C) LLM capped at 3s
      "llm timeout"
    );
    const j = await r.json();
    const txt = j?.choices?.[0]?.message?.content || "";
    const m = txt.match(/\{[\s\S]*\}$/);
    if(!m) return { recipe:null, llmError:"no JSON in completion" };

    const parsed = JSON.parse(m[0]);
    parsed.id = parsed.id || `llm-${Date.now()}`;
    parsed.energy = parsed.energy || "hob";
    parsed.cost = parsed.cost ?? 2.0;
    parsed.badges = Array.isArray(parsed.badges) ? parsed.badges : ["llm"];
    parsed.ingredients = (parsed.ingredients||[]).map(i => {
      const nm = String(i.name||"").toLowerCase();
      return { name: nm, have: pantry.includes(nm) };
    });
    parsed.steps = (parsed.steps||[]).map((s,i)=>({ id: s.id || `step-${i}`, text: s.text || String(s) }));
    return { recipe: parsed };
  }catch(e){
    return { recipe:null, llmError:String(e?.message||e) };
  }
}

// ---------- Emergency local (never-empty) ----------
function emergencyRecipe(pantry){
  const titleBits = [];
  if (pantry.some(p => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (pantry.includes("coconut cream")) titleBits.push("Coconut");
  if (pantry.some(p => PASTA_WORDS.has(p))) titleBits.push("Spaghetti");
  const title = (titleBits.length ? titleBits.join(" ") : "Pantry") + " Savoury Skillet";

  const ings = [
    ...pantry.map(p => ({ name: p, have: true })),
    { name:"onion (or onion powder)", have:false },
    { name:"garlic (or garlic powder)", have:false },
    { name:"ginger (or ground ginger)", have:false },
    { name:"mixed dried herbs", have:false },
    { name:"stock cube", have:false },
    { name:"lemon or vinegar", have:false },
    { name:"salt & black pepper", have:false }
  ];
  const steps = [
    { id:"s1", text:"Heat oil; add onion, garlic & ginger. Cook 2–3 min." },
    { id:"s2", text:"Add pantry items (e.g., chickpeas, coconut cream, any chopped veg). Stir." },
    { id:"s3", text:"Season with herbs, stock, salt & pepper; add splash of pasta water if using spaghetti." },
    { id:"s4", text:"Simmer 6–8 min to thicken. Toss with cooked spaghetti or serve over rice." }
  ];
  return { id:`local-${Date.now()}`, title, time:15, cost:2.0, energy:"hob", ingredients:ings, steps, badges:["local"] };
}

// ---------- handler ----------
export default async function handler(req){
  if (req.method !== "POST") return json(405,{error:"POST only"});
  let body; try{ body = await req.json(); } catch{ return json(400,{error:"invalid JSON"}); }

  const { imageBase64, pantryOverride, prefs = {} } = body || {};
  let pantry = [], source = "", pantryFrom = {};

  // Source selection + Vision with timeout
  if (Array.isArray(pantryOverride) && pantryOverride.length){
    pantry = cleanPantry(pantryOverride);
    source = "pantryOverride";
  } else if (imageBase64){
    try {
      const res = await withTimeout(
        () => callVision(imageBase64),
        3500,
        "vision timeout"
      );
      pantryFrom = { ocr: res.ocrTokens, labels: res.labels, objects: res.objects };
      const combined = [...(res.ocrTokens||[]), ...(res.labels||[]), ...(res.objects||[])];
      pantry = cleanPantry(combined);
      source = "vision";
    } catch (e) {
      // Vision timed out/failed → still let user continue; pantry will be empty here
      pantry = cleanPantry([]);
      source = "vision-failed";
      pantryFrom = { error: String(e?.message||e) };
    }
  } else {
    return json(400,{error:"imageBase64 required (or pantryOverride)"});
  }

  // SPOONACULAR (2.5s)
  const sp = await spoonacularRecipes(pantry, prefs);
  let recipes = sp.results || [];

  // LLM fallback (3s) if Spoon returned none (or got filtered to none)
  let usedLLM = false, llmTried = false, llmError = null, reasonNoLLM = null;
  if (recipes.length === 0) {
    llmTried = true;
    const llm = await llmRecipe(pantry, prefs);
    if (llm.reasonNoLLM) reasonNoLLM = llm.reasonNoLLM;
    if (llm.llmError)    llmError    = llm.llmError;
    if (llm.recipe){ recipes = [llm.recipe]; usedLLM = true; }
  }

  // NEVER return empty – hand back emergency recipe
  if (recipes.length === 0){
    recipes = [emergencyRecipe(pantry)];
  }

  const debug = {
    source,
    pantryFrom,
    cleanedPantry: pantry,
    extra: { spoonacularRawCount: sp.extra?.spoonacularRawCount, kept: sp.extra?.kept, timeCap: sp.extra?.timeCap },
    usedLLM, llmTried, llmError, reasonNoLLM
  };

  console.log("analyze debug:", debug);
  return json(200, { pantry, recipes, debug });
}
