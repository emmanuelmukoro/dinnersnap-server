// api/analyze.js
// Node.js Serverless (req, res) version to avoid "Illegal return statement".
// Fast, resilient pipeline: Vision → cleaned pantry → (Spoonacular ⟂ LLM) → emergency fallback.

export const config = {
  runtime: "nodejs",
  maxDuration: 25,
  memory: 1024,
  regions: ["lhr1"],
};

/* ---------------- ENV KEYS ---------------- */
const GCV_KEY        = process.env.GCV_KEY;
const SPOON_KEY      = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

/* ---------------- Utils ---------------- */
function send(res, status, obj) {
  res.status(status);
  res.setHeader("content-type", "application/json");
  res.send(JSON.stringify(obj));
}

function nowMs() { return Number(process.hrtime.bigint() / 1000000n); }

function withTimeout(run, ms, label = "timeout") {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ms);
  const raced = Promise.race([
    run(controller.signal),
    new Promise((_, rej) => setTimeout(() => rej(new Error(label)), ms + 10)),
  ]);
  return raced.finally(() => clearTimeout(timeout));
}

function tfetch(url, opts = {}, ms = 2500, label = "fetch-timeout") {
  return withTimeout((signal) => fetch(url, { ...opts, signal }), ms, label);
}

/* ---------------- Classifiers ---------------- */
const DESSERT = new Set(["dessert","pudding","ice cream","smoothie","shake","cookie","brownie","cupcake","cake","muffin","pancake","waffle","jam","jelly","compote","truffle","fudge","sorbet","parfait","custard","tart","shortbread","licorice","cobbler"]);
const DRINK   = new Set(["drink","beverage","mocktail","cocktail","margarita","mojito","spritzer","soda","punch","toddy"]);
const ALCOHOL = new Set(["brandy","rum","vodka","gin","whisky","whiskey","bourbon","tequila","wine","liqueur","amaretto","cognac","port","sherry"]);
const GENERIC = new Set(["pasta","rice","bread","flour","sugar","oil","salt","pepper","milk","seasoning","spices"]);
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

/* ---------------- Google Vision (2.5s cap) ---------------- */
async function callVision(imageBase64) {
  // If key missing, THROW so handler logs vision-failed explicitly.
  if (!GCV_KEY) throw new Error("GCV_KEY missing");

  const body = {
    requests: [{
      image: { content: imageBase64.replace(/^data:image\/\w+;base64,/, "") },
      features: [
        { type: "TEXT_DETECTION",       maxResults: 1  },
        { type: "LABEL_DETECTION",      maxResults: 10 },
        { type: "OBJECT_LOCALIZATION",  maxResults: 10 },
      ],
    }],
  };

  const r = await tfetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${GCV_KEY}`,
    { method:"POST", headers:{ "content-type":"application/json" }, body: JSON.stringify(body) },
    2500,
    "vision-timeout"
  );
  if (!r.ok) throw new Error(`vision http ${r.status}`);

  const j = await r.json();
  if (j?.error) throw new Error(`vision api: ${j.error.message || "unknown error"}`);

  const res = j?.responses?.[0] || {};
  const rawText = (res.textAnnotations?.[0]?.description || "").toLowerCase();
  const RAW_TOKENS = rawText.split(/[^a-z]+/g).map(t => t.trim()).filter(Boolean);
  const FOODISH = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh","frozen",
    "beans","peas","lentils","broccoli","banana","tomato","onion","garlic","squash",
    "chickpea","chickpeas","feta","rice","pasta","spaghetti","oats","milk","yogurt",
    "cheese","lemon","orange","pepper","spinach","mushroom","potato","avocado","coconut","cream",
  ]);
  const ocrTokens = RAW_TOKENS.filter(t => t.length >= 4 && FOODISH.has(t));
  const labels  = (res.labelAnnotations || []).map(x => x.description?.toLowerCase()).filter(Boolean);
  const objects = (res.localizedObjectAnnotations || []).map(x => x.name?.toLowerCase()).filter(Boolean);
  return { ocrTokens, labels, objects };
}

/* ---------------- Pantry cleanup ---------------- */
const WHITELIST = [
  "banana","broccoli","chickpeas","tomatoes","onion","garlic","olive oil","pasta","spaghetti",
  "courgette","feta","orzo","rice","egg","spring onion","lemon","orange","avocado","coconut cream","coconut milk",
  "butternut squash","squash","carrot","pepper","bell pepper","spinach",
  "potato","mushroom","cheese","yogurt","milk","chicken","chicken breast","beef","pork","fish","salmon","tuna",
  "bread","tortilla","wrap","beans","kidney beans","black beans","lentils",
  "cucumber","lettuce","cabbage","kale","apple","pear","oats","flour","sugar","butter","salt","pepper""almond milk",
"jeera",
"cumin",
"cloves",
"maggi seasoning",
"masala",
"garam masala",
"scotch bonnet",
"plantain",
"callaloo",
"kidney beans",
"coconut milk",
"curry powder"
];

const MAP = {
  bananas:"banana", chickpea:"chickpeas", garbanzo:"chickpeas",
  "garbanzo bean":"chickpeas", "garbanzo beans":"chickpeas",
  zucchini:"courgette", courgettes:"courgette", tomato:"tomatoes",
  onions:"onion", eggs:"egg", brockley:"broccoli",
  "red kidney beans":"kidney beans", "coconutmilk":"coconut milk", "beef seasoning":"seasoning", "maggi": "maggi seasoning",
"jeera": "cumin"
};

const uniqLower = (arr) => { const s=new Set(); for (const x of arr) s.add(String(x).toLowerCase()); return [...s]; };
function lev(a,b){const m=[];for(let i=0;i<=b.length;i++)m[i]=[i];for(let j=0;j<=a.length;j++)m[0][j]=j;for(let i=1;i<=b.length;i++){for(let j=1;j<=a.length;j++){m[i][j]=Math.min(m[i-1][j]+1,m[i][j-1]+1,m[i-1][j-1]+(b.charAt(i-1)===a.charAt(j-1)?0:1));}}return m[b.length][a.length];}
function nearestWhitelistStrict(term){let best=null,bestDist=2;for(const w of WHITELIST){const d=lev(term,w);if(d<bestDist){bestDist=d;best=w;}}return bestDist<=1?best:null;}

/** Join OCR tokens into food phrases (e.g., "coconut milk", "kidney beans"). */
function joinFoodPhrases(tokens) {
  const set = new Set(tokens.map(t => t.toLowerCase()));
  const out = new Set(tokens);
  if (set.has("coconut") && set.has("milk")) out.add("coconut milk");
  if (set.has("kidney") && set.has("beans")) out.add("kidney beans");
  if (set.has("red") && set.has("kidney") && set.has("beans")) out.add("kidney beans");
  return [...out];
}

function cleanPantry(raw){
  const out=new Set();
  for(const t0 of uniqLower(raw)){
    const t = MAP[t0] || t0;
    if (WHITELIST.includes(t)) { out.add(t); continue; }
    if (t.length>=5){
      const near=nearestWhitelistStrict(t);
      if (near) { out.add(near); continue; }
    }
  }
  return [...out];
}

/* ---------------- Spoonacular (2.0s cap) ---------------- */
async function spoonacularRecipes(pantry, prefs){
  if (!SPOON_KEY) return { results: [], info: { note: "no SPOON_KEY" } };
  const hasMeat = pantry.some(p => MEAT.has(p));
  const include = pantry.join(",");
  const timeCap = Math.max(10, (prefs?.time ?? 25) + 10);

  const url = new URL("https://api.spoonacular.com/recipes/complexSearch");
  url.searchParams.set("apiKey", SPOON_KEY);
  url.searchParams.set("includeIngredients", include);
  url.searchParams.set("instructionsRequired","true");
  url.searchParams.set("addRecipeInformation","true");
  url.searchParams.set("sort","max-used-ingredients");

if (Number(prefs?.explore) > 0) {
   url.searchParams.set("sort","random");
   // random offset up to a small range to vary hits but stay fast
   const offset = Math.min(100, (prefs.explore % 5) * 20);
   url.searchParams.set("offset", String(offset));
 }


  url.searchParams.set("number","24"); // a touch more
  url.searchParams.set("ignorePantry","true");
  url.searchParams.set("type","main course");
  url.searchParams.set("excludeIngredients", Array.from(ALCOHOL).join(","));
  url.searchParams.set("maxReadyTime", String(timeCap));
  if (!hasMeat) url.searchParams.set("diet","vegetarian");

  let j={};
  try {
    const r = await tfetch(url.toString(), {}, 2000, "spoon-timeout");
    j = await r.json();
  } catch(e){
    return { results: [], info:{ error:String(e?.message||e) } };
  }

  const raw  = Array.isArray(j?.results) ? j.results : [];
  const spec = pantry.filter(p => !GENERIC.has(p));

  const filtered = raw.filter(it => {
    const title = String(it.title||"").toLowerCase();
    const dish  = (it.dishTypes||[]).map(d => String(d).toLowerCase());
    if (titleHasAny(title, DESSERT)) return false;
    if (titleHasAny(title, DRINK))   return false;
    if (titleHasAny(title, ALCOHOL)) return false;
    if (dish.some(d => ["drink","beverage","cocktail","dessert"].includes(d))) return false;

    const used   = it.usedIngredientCount ?? 0;
    const missed = it.missedIngredientCount ?? 0;
    const time   = it.readyInMinutes ?? 999;
    if (used < 1) return false;      // allow 1 used to avoid over-pruning
    if (time > timeCap) return false;

    if (spec.length >= 2 && !hasSpecificTermInTitleOrIngredients(it, spec)) return false;

    if (title.includes("macaroni and cheese")) {
      const hasDairy = pantry.some(p => ["cheese","milk","cream","yogurt"].includes(p));
      if (!hasDairy) return false;
    }
    if ((/shrimp|prawn|salmon|tuna|anchovy|sardine/).test(title) &&
        !pantry.some(p => ["shrimp","prawn","salmon","tuna","fish"].includes(p))) return false;

    return true;
  });

  const scored = filtered.map(it => {
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
      ingredients: (it.extendedIngredients||[]).map(ing => {
        const nm = String(ing.name||"").toLowerCase();
        return { name: nm, have: pantry.includes(nm) };
      }),
      steps: ((it.analyzedInstructions?.[0]?.steps)||[]).map((s,i)=>({ id:`${it.id}-s${i}`, text:s.step })),
      badges: ["web"],
    };
  });

  return { results: scored, info:{ spoonacularRawCount: raw.length, kept: scored.length, timeCap } };
}

/* ---------------- LLM (3.2s cap) ---------------- */
function buildPrompt(pantry, prefs){
  const core     = pantry.join(", ");
  const minutes  = Math.min(Math.max(prefs?.time || 25, 10), 40);
  const servings = Math.min(Math.max(prefs?.servings || 2, 1), 6);
  const diet     = prefs?.diet || "none";
  const energy   = prefs?.energyMode || "hob";


 const exploreNote = Number(prefs?.explore) > 0 ? "Vary style/cuisine a bit compared to earlier suggestions." : "";
 return `


Create ONE savoury DINNER recipe (no drinks, no desserts) using primarily: ${core}.
Assume staples: salt, pepper, oil, stock cube/paste, onion, garlic, chilli, ginger, smoked paprika,
curry powder, dried herbs, soy sauce, lemon/lime, vinegar.
Ensure flavour: aromatics (onion/garlic/ginger), an acid/freshness (lemon/lime/soy),
and at least 3 seasonings from the staples list if missing from pantry.
Constraints:
- Ready in <= ${minutes} minutes
- Servings: ${servings}
- Diet: ${diet}
- Prefers: ${energy}
Return ONLY JSON with:
{
  "id": "llm-<id>", "title": "...", "time": ${minutes}, "cost": 3, "energy": "${energy}",
  "ingredients": [{"name":"...", "have": true|false}], "steps": [{"id":"s1","text":"..."}],
  "badges": ["under-30","budget"]
}`;
}

async function llmRecipe(pantry, prefs){
  if (!OPENAI_API_KEY) return { recipe:null, info:{ reason:"no OPENAI_API_KEY" } };

  const body = {
    model: "gpt-4o-mini",
    temperature: 0.45,
    max_tokens: 600,
    messages: [
      { role:"system", content:"You write concise, savoury DINNER recipes only." },
      { role:"user",   content: buildPrompt(pantry, prefs) }
    ]
  };

  try {
    const r = await tfetch(
      "https://api.openai.com/v1/chat/completions",
      { method:"POST", headers:{ "content-type":"application/json", authorization:`Bearer ${OPENAI_API_KEY}` }, body: JSON.stringify(body) },
      3200, // was 2500
      "llm-timeout"
    );
    const j   = await r.json();
    const txt = j?.choices?.[0]?.message?.content || "";
    const m   = txt.match(/\{[\s\S]*\}$/);
    if (!m) return { recipe:null, info:{ error:"no JSON in completion" } };

    const parsed = JSON.parse(m[0]);
    parsed.id        = parsed.id || `llm-${Date.now()}`;
    parsed.energy    = parsed.energy || "hob";
    parsed.cost      = parsed.cost ?? 2.0;
    parsed.badges    = Array.isArray(parsed.badges) ? parsed.badges : ["llm"];
    parsed.ingredients = (parsed.ingredients||[]).map(i => {
      const nm = String(i.name||"").toLowerCase();
      return { name: nm, have: pantry.includes(nm) };
    });
    parsed.steps = (parsed.steps||[]).map((s,i)=>({ id: s.id || `step-${i}`, text: s.text || String(s) }));
    return { recipe: parsed, info:{ ok:true } };
  } catch(e){
    return { recipe:null, info:{ error:String(e?.message||e) } };
  }
}

/* ---------------- Emergency (instant) ---------------- */
function emergencyRecipe(pantry){
  const have = (n) => pantry.includes(n);
  const titleBits = [];
  if (pantry.some(p => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (pantry.includes("coconut cream") || pantry.includes("coconut milk")) titleBits.push("Coconut");
  if (pantry.some(p => PASTA.has(p))) titleBits.push("Spaghetti");
  const title = (titleBits.length ? titleBits.join(" ") : "Pantry") + " Savoury Skillet";

  const ings = [
    ...pantry.map(p => ({ name:p, have:true })),
    ...(!have("onion")  ? [{ name:"onion (or onion powder)",  have:false }] : []),
    ...(!have("garlic") ? [{ name:"garlic (or garlic powder)",have:false }] : []),
    ...(!have("ginger") ? [{ name:"ginger (or ground ginger)",have:false }] : []),
    { name:"mixed dried herbs",        have:false },
    { name:"stock cube",               have:false },
    { name:"lemon or vinegar",         have:false },
    { name:"salt & black pepper",      have:false },
  ];

  const steps = [
    { id:"s0", text:"Put a pan of salted water on to boil (for pasta) or start rice." },
    { id:"s1", text:"Heat oil in a skillet; add onion, garlic & ginger. Cook 2–3 min." },
    { id:"s2", text:"Add pantry items (e.g., chickpeas, coconut milk/cream, chopped veg). Stir." },
    { id:"s3", text:"Season with herbs, stock, salt & pepper; if using pasta, cook it now until al dente." },
    { id:"s4", text:"Simmer sauce 6–8 min. Toss with drained pasta (use a splash of pasta water) or serve over rice." },
  ];

  return { id:`local-${Date.now()}`, title, time:15, cost:2.0, energy:"hob", ingredients:ings, steps, badges:["local"] };
}

/* ---------------- Handler (Node.js req, res) ---------------- */
export default async function handler(req, res) {
  const t0 = nowMs();
  if (req.method !== "POST") return send(res, 405, { error: "POST only" });

  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch { /* framework may parse already */ }
  if (!Object.keys(body).length && typeof req.body !== "string") {
    body = req.body || {};
  }

  const { imageBase64, pantryOverride, prefs = {} } = body || {};
  const watchdogMs = 8000;

  const main = async () => {
    const tStart = nowMs();

    // Step 1: Build pantry
    let pantry = [];
    let source = "";
    let pantryFrom = {};

    if (Array.isArray(pantryOverride) && pantryOverride.length) {
      pantry = cleanPantry(pantryOverride);
      source = "pantryOverride";
    } else if (imageBase64) {
      try {
        const v0 = nowMs();
        const resV = await withTimeout(() => callVision(imageBase64), 2500, "vision-timeout");
        pantryFrom = { ocr: resV.ocrTokens, labels: resV.labels, objects: resV.objects };
        const combined = joinFoodPhrases([...(resV.ocrTokens||[]), ...(resV.labels||[]), ...(resV.objects||[])]);
        pantry = cleanPantry(combined);
        source = "vision";
        console.log(`vision ok in ${nowMs()-v0}ms, pantry=`, pantry);
      } catch(e){
        source = "vision-failed";
        pantryFrom = { error: String(e?.message||e) };
        pantry = []; // continue with downstream providers
        console.log("vision error:", pantryFrom.error);
      }
    } else {
      // allow emergency/LLM path so "Find recipes" works even with empty detected list
      source = "no-input";
      pantry = [];
    }

    // Step 2: kick off both providers in parallel
    const sPromise = (async () => {
      const s0 = nowMs();
      const sp = await spoonacularRecipes(pantry, prefs);
      console.log(`spoon end in ${nowMs()-s0}ms kept=${sp.info?.kept} raw=${sp.info?.spoonacularRawCount}`);
      return Array.isArray(sp.results) ? sp.results : [];
    })();

    const lPromise = (async () => {
      const l0 = nowMs();
      const llm = await llmRecipe(pantry, prefs);
      console.log(`llm end in ${nowMs()-l0}ms ok=${!!llm.recipe} err=${llm.info?.error || llm.info?.reason || ""}`);
      return llm.recipe || null;
    })();

    // Step 3: race for primary, then merge with whatever else finished
    let primary = null;
    let usedLLM = false;
    try {
      const winner = await Promise.race([
        sPromise.then(arr => ({ tag:"spoon", arr })).catch(() => ({ tag:"spoon", arr:[] })),
        lPromise.then(rec => ({ tag:"llm",   rec })).catch(() => ({ tag:"llm",   rec:null })),
      ]);
      if (winner.tag === "spoon" && Array.isArray(winner.arr) && winner.arr.length) {
        primary = winner.arr[0];
      } else if (winner.tag === "llm" && winner.rec) {
        primary = winner.rec;
        usedLLM = true;
      }
    } catch {}

    const [s2, l2] = await Promise.allSettled([sPromise, lPromise]);
    const spoonArr = s2.status === "fulfilled" && Array.isArray(s2.value) ? s2.value : [];
    const llmOne   = l2.status === "fulfilled" && l2.value ? [l2.value] : [];

    const pool = [];
    if (primary) pool.push(primary);
    pool.push(...spoonArr, ...llmOne);
    const seen = new Set();
    const recipes = [];
    for (const r of pool) {
      if (!r || !r.id) continue;
      const id = String(r.id);
      if (seen.has(id)) continue;
      seen.add(id);
      recipes.push(r);
      if (recipes.length >= 8) break;
    }

    // Never empty; and ensure we have alternates
    if (recipes.length === 0) recipes.push(emergencyRecipe(pantry));
    if (recipes.length < 3) {
      const alts = [];
      if (pantry.some(p => PASTA.has(p))) {
        alts.push({
          id:`alt-${Date.now()}-aglio`,
          title:"Aglio e Olio (Pantry Fast)",
          time:12, cost:1.2, energy:"hob", badges:["local","under-15"],
          ingredients:[
            { name:"spaghetti", have: pantry.some(p=>PASTA.has(p)) },
            { name:"garlic", have: pantry.includes("garlic") },
            { name:"chilli or chilli flakes", have:false },
            { name:"olive oil", have: pantry.includes("olive oil") },
            { name:"salt", have:false }, { name:"black pepper", have:false }
          ],
          steps:[
            {id:"s1",text:"Boil salted water; cook spaghetti until al dente."},
            {id:"s2",text:"Meanwhile warm oil; sizzle sliced garlic & chilli 30–60s (don’t brown)."},
            {id:"s3",text:"Toss pasta with garlicky oil; season. Add splash of pasta water to gloss."}
          ]
        });
      }
      if (pantry.includes("kidney beans") || pantry.includes("chickpeas")) {
        alts.push({
          id:`alt-${Date.now()}-bean`,
          title:"Speedy Spiced Beans",
          time:15, cost:1.5, energy:"hob", badges:["local","budget"],
          ingredients:[
            {name:"kidney beans or chickpeas", have: pantry.includes("kidney beans") || pantry.includes("chickpeas")},
            {name:"onion", have: pantry.includes("onion")},
            {name:"garlic", have: pantry.includes("garlic")},
            {name:"mixed dried herbs or curry powder", have:false},
            {name:"stock cube", have:false}
          ],
          steps:[
            {id:"s1",text:"Sauté onion & garlic in oil 2–3 min."},
            {id:"s2",text:"Add beans, spices, crumble in stock; splash water; simmer 8–10 min."},
            {id:"s3",text:"Finish with lemon or vinegar; serve with rice/toast/wraps."}
          ]
        });
      }
      for (const r of alts) { if (recipes.length >= 3) break; recipes.push(r); }
    }

    const debug = {
      source,
      pantryFrom,
      cleanedPantry: pantry,
      usedLLM,
      totalMs: nowMs() - tStart,
    };
    console.log("analyze debug:", debug);
    return send(res, 200, { pantry, recipes, debug });
  };

  try {
    await withTimeout(() => main(), watchdogMs, "watchdog-timeout");
    console.log(`handler total=${nowMs()-t0}ms`);
  } catch(e){
    console.log("watchdog fired:", String(e?.message||e));
    return send(res, 200, {
      pantry: [],
      recipes: [emergencyRecipe([])],
      debug: { source: "watchdog", error: String(e?.message||e) },
    });
  }
}
