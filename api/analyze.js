// api/analyze.js
// Same pipeline, but Node-style (req, res) so Vercel won't hang on req.json().

export const config = {
  runtime: "nodejs",
  maxDuration: 25,
  memory: 1024,
  regions: ["lhr1"],
};

/* ----------------------- Environment Keys ----------------------- */
const GCV_KEY = process.env.GCV_KEY;
const SPOON_KEY = process.env.SPOON_KEY || process.env.SPOONACULAR_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

/* ----------------------- Basic Helpers ----------------------- */
function nowMs() {
  return Number(process.hrtime.bigint() / 1000000n);
}

function withTimeout(run, ms, label = "timeout") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return Promise.race([
    run(controller.signal),
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(label)), ms + 10)
    ),
  ]).finally(() => clearTimeout(timer));
}

function tfetch(url, opts = {}, ms = 2500, label = "fetch-timeout") {
  return withTimeout(
    (signal) => fetch(url, { ...opts, signal }),
    ms,
    label
  );
}

async function withRetry(fn, tries = 2) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); }
    catch (e) { last = e; }
  }
  throw last;
}




/* ----------------------- Classification Sets ----------------------- */
const DESSERT = new Set([
  "dessert","pudding","ice cream","smoothie","shake","cookie","brownie",
  "cupcake","cake","muffin","pancake","waffle","jam","jelly","compote",
  "truffle","fudge","sorbet","parfait","custard","tart","shortbread",
  "licorice","cobbler",
]);
const DRINK = new Set([
  "drink","beverage","mocktail","cocktail","margarita","mojito","spritzer",
  "soda","punch","toddy",
]);
const ALCOHOL = new Set([
  "brandy","rum","vodka","gin","whisky","whiskey","bourbon","tequila","wine",
  "liqueur","amaretto","cognac","port","sherry",
]);
const GENERIC = new Set([
  "pasta","rice","bread","flour","sugar","oil","salt","pepper",
]);
const MEAT = new Set([
  "chicken","beef","pork","lamb","bacon","ham","turkey","shrimp",
  "prawn","salmon","tuna","fish",
]);
const PASTA = new Set([
  "pasta","spaghetti","macaroni","penne","farfalle","orzo","fusilli",
  "linguine","tagliatelle",
]);

const titleHasAny = (title, set) => {
  const t = String(title || "").toLowerCase();
  for (const w of set) if (t.includes(w)) return true;
  return false;
};

const hasSpecificTermInTitleOrIngredients = (item, specificTerms) => {
  const t = String(item.title || "").toLowerCase();
  const ings = (item.extendedIngredients || []).map((i) =>
    String(i.name || "").toLowerCase()
  );
  return specificTerms.some(
    (p) => t.includes(p) || ings.some((n) => n.includes(p))
  );
};

/* ----------------------- Vision (2.5s cap) ----------------------- */
async function callVision(imageBase64) {
  if (!GCV_KEY) {
    return { ocrTokens: [], labels: [], objects: [], error: "no GCV_KEY" };
  }

  const body = {
    requests: [
      {
        image: {
          content: imageBase64.replace(/^data:image\/\w+;base64,/, ""),
        },
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
  const RAW_TOKENS = rawText
    .split(/[^a-z]+/g)
    .map((t) => t.trim())
    .filter(Boolean);

  const FOODISH = new Set([
    "ingredients","organic","sauce","soup","canned","tin","dried","fresh",
    "frozen","beans","peas","lentils","broccoli","banana","tomato","onion",
    "garlic","squash","chickpea","chickpeas","feta","rice","pasta","spaghetti",
    "oats","milk","yogurt","cheese","lemon","orange","pepper","spinach",
    "mushroom","potato","avocado","coconut","cream",
  ]);

  const ocrTokens = RAW_TOKENS.filter(
    (t) => t.length >= 4 && FOODISH.has(t)
  );

  const labels = (res.labelAnnotations || [])
    .map((x) => x.description?.toLowerCase())
    .filter(Boolean);

  const objects = (res.localizedObjectAnnotations || [])
    .map((x) => x.name?.toLowerCase())
    .filter(Boolean);

  return { ocrTokens, labels, objects };
}

/* ----------------------- Pantry cleanup ----------------------- */
const WHITELIST = [
  "banana","broccoli","chickpeas","beans","kidney beans","tomatoes","onion",
  "garlic","ginger","olive oil","pasta","spaghetti","courgette","feta","orzo",
  "rice","egg","spring onion","lemon","orange","avocado","coconut milk",
  "coconut cream","butternut squash","squash","carrot","pepper","bell pepper",
  "spinach","potato","mushroom","cheese","cheddar","yogurt","milk",
  "almond milk","chicken","chicken breast","beef","pork","fish","salmon",
  "tuna","bread","tortilla","wrap","lentils","cucumber","lettuce","cabbage",
  "kale","apple","pear","oats","flour","sugar","butter","salt","pepper",
  "stock cube","curry powder","mixed dried herbs","garam masala",
  "maggi seasoning","jeera","cumin","cloves","all purpose seasoning", "steak seasoning",
 "beef seasoning",
 "paprika",
 "smoked paprika",
 "curry paste",
 "soy sauce",
 "tropical sun",
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
  "kidney beans": "kidney beans",
  "coconut milk": "coconut milk",
  "coconut cream": "coconut cream",
  maggi: "maggi seasoning",
  "jeera powder": "jeera",
};

const uniqLower = (arr) => {
  const s = new Set();
  for (const x of arr) s.add(String(x).toLowerCase());
  return [...s];
};

function lev(a, b) {
  const m = [];
  for (let i = 0; i <= b.length; i++) m[i] = [i];
  for (let j = 0; j <= a.length; j++) m[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      m[i][j] = Math.min(
        m[i - 1][j] + 1,
        m[i][j - 1] + 1,
        m[i - 1][j - 1] + (b.charAt(i - 1) === a.charAt(j - 1) ? 0 : 1)
      );
    }
  }
  return m[b.length][a.length];
}

function nearestWhitelistStrict(term) {
  let best = null;
  let bestDist = 2;
  for (const w of WHITELIST) {
    const d = lev(term, w);
    if (d < bestDist) {
      bestDist = d;
      best = w;
    }
  }
  return bestDist <= 1 ? best : null;
}

function cleanPantry(raw) {
  const out = new Set();
  for (const t0 of uniqLower(raw)) {
    const t = MAP[t0] || t0;
    if (WHITELIST.includes(t)) {
      out.add(t);
      continue;
    }
    if (t.length >= 5) {
      const near = nearestWhitelistStrict(t);
      if (near) {
        out.add(near);
        continue;
      }
    }
  }
  return [...out];
}

/* ----------------------- Spoonacular (2.0s cap) ----------------------- */
async function spoonacularRecipes(pantry, prefs) {
  if (!SPOON_KEY) return { results: [], info: { note: "no SPOON_KEY" } };

  const hasMeat = pantry.some((p) => MEAT.has(p));
  const include = pantry.join(",");
  const timeCap = Math.max(10, (prefs?.time ?? 25) + 10);

  const url = new URL(
    "https://api.spoonacular.com/recipes/complexSearch"
  );
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
  if (!hasMeat) {
    url.searchParams.set("diet", "vegetarian");
  }

  let j = {};
  try {
    const r = await withRetry(() => tfetch(url.toString(), {}, 2000, "spoon-timeout"));
    j = await r.json();
  } catch (e) {
    return { results: [], info: { error: String(e?.message || e) } };
  }

  const raw = Array.isArray(j?.results) ? j.results : [];
  const specific = pantry.filter((p) => !GENERIC.has(p));

  // strict filter first
const filtered = raw.filter((it) => {
  const title = String(it.title || "").toLowerCase();
  const dish = (it.dishTypes || []).map((d) => String(d).toLowerCase());

  if (titleHasAny(title, DESSERT)) return false;
  if (titleHasAny(title, DRINK)) return false;
  if (titleHasAny(title, ALCOHOL)) return false;
  if (dish.some((d) => ["drink", "beverage", "cocktail", "dessert"].includes(d))) return false;

  const used = it.usedIngredientCount ?? 0;
  const time = it.readyInMinutes ?? 999;

  if (used < 2) return false;
  if (time > timeCap) return false;

  // only enforce “specific term” rule when we actually have >= 2 specific pantry items
  if (specific.length >= 2 && !hasSpecificTermInTitleOrIngredients(it, specific)) return false;

  // fish sanity check
  if (
    /shrimp|prawn|salmon|tuna|anchovy|sardine/.test(title) &&
    !pantry.some((p) => ["shrimp", "prawn", "salmon", "tuna", "fish"].includes(p))
  ) return false;

  return true;
});

// if nothing survived strict filter, try a relaxed one
let survivors = filtered;
if (survivors.length === 0) {
  const relaxed = raw.filter((it) => {
    const title = String(it.title || "").toLowerCase();
    const dish = (it.dishTypes || []).map((d) => String(d).toLowerCase());

    if (titleHasAny(title, DESSERT) || titleHasAny(title, DRINK) || titleHasAny(title, ALCOHOL)) return false;
    if (dish.some((d) => ["drink","beverage","cocktail","dessert"].includes(d))) return false;

    const used = it.usedIngredientCount ?? 0;
    const time = it.readyInMinutes ?? 999;

    // relax used>=1, keep time cap
    if (used < 1) return false;
    if (time > timeCap) return false;

    return true;
  });
  survivors = relaxed;
}

// score + shape
const scored = survivors.map((it) => {
  const used = it.usedIngredientCount ?? 0;
  const missed = it.missedIngredientCount ?? 0;
  const time = it.readyInMinutes ?? 30;
  const score = 0.7 * (used / (used + missed + 1)) + 0.3 * (1 - Math.min(time, 60) / 60);

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

// shuffle a touch if explore
const arr = [...scored];
if (prefs.explore) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j2 = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j2]] = [arr[j2], arr[i]];
  }
}

return {
  results: arr,
  info: {
    spoonacularRawCount: raw.length,
    kept: arr.length,
    timeCap,
  },
};


/* ----------------------- LLM (2.5s cap) ----------------------- */
function buildPrompt(pantry, prefs) {
  const core = pantry.join(", ");
  const minutes = Math.min(Math.max(prefs?.time || 25, 10), 40);
  const servings = Math.min(Math.max(prefs?.servings || 2, 1), 6);
  const diet = prefs?.diet || "none";
  const energy = prefs?.energyMode || "hob";

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
  "id": "llm-<id>",
  "title": "...",
  "time": ${minutes},
  "cost": 3,
  "energy": "${energy}",
  "ingredients": [{"name":"...", "have": true|false}],
  "steps": [{"id":"s1","text":"..."}],
  "badges": ["under-30","budget"]
}`;
}

async function llmRecipe(pantry, prefs) {
  if (!OPENAI_API_KEY)
    return { recipe: null, info: { reason: "no OPENAI_API_KEY" } };

  const body = {
    model: "gpt-4o-mini",
    temperature: 0.45,
    max_tokens: 600,
    messages: [
      {
        role: "system",
        content: "You write concise, savoury DINNER recipes only.",
      },
      { role: "user", content: buildPrompt(pantry, prefs) },
    ],
  };

  try {
    const r = await withRetry( () => tfetch("https://api.openai.com/v1/chat/completions", { ... }, 4000, "llm-timeout")
 );

    const j = await r.json();
    const txt = j?.choices?.[0]?.message?.content || "";
    const m = txt.match(/\{[\s\S]*\}$/);
    if (!m)
      return { recipe: null, info: { error: "no JSON in completion" } };

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

/* ----------------------- Emergency Recipe ----------------------- */
function emergencyRecipe(pantry, explore = 0) {
  const titleBits = [];
  if (pantry.some((p) => p.includes("chickpea"))) titleBits.push("Chickpea");
  if (pantry.includes("coconut cream")) titleBits.push("Coconut");
  if (pantry.some((p) => PASTA.has(p))) titleBits.push("Spaghetti");

  const title =
    (titleBits.length ? titleBits.join(" ") : "Pantry") + ` Savoury Skillet (idea ${((explore % 5) + 1)})`;

  const baseIngs = [
    { name: "onion (or onion powder)", have: pantry.includes("onion") },
    { name: "garlic (or garlic powder)", have: pantry.includes("garlic") },
    { name: "ginger (or ground ginger)", have: pantry.includes("ginger") },
    {
      name: "mixed dried herbs",
      have: pantry.includes("mixed dried herbs"),
    },
    {
      name: "stock cube",
      have: pantry.includes("stock cube"),
    },
    {
      name: "lemon or vinegar",
      have: pantry.includes("lemon") || pantry.includes("vinegar"),
    },
    {
      name: "salt & black pepper",
      have: pantry.includes("salt") || pantry.includes("black pepper"),
    },
  ];

  const uniqPantry = pantry.map((p) => ({ name: p, have: true }));

  return {
    id: `local-${Date.now()}-${explore}`,
    title,
    time: 15,
    cost: 2.0,
    energy: "hob",
    ingredients: [...uniqPantry, ...baseIngs],
    steps: [
      {
        id: "s1",
        text: "Heat oil; add onion, garlic & ginger. Cook 2–3 min.",
      },
      {
        id: "s2",
        text: "Add pantry items (e.g., chickpeas, coconut cream, chopped veg). Stir.",
      },
      {
        id: "s3",
        text: "Season with herbs, stock, salt & pepper; loosen with splash of pasta water if using spaghetti.",
      },
      {
        id: "s4",
        text: "Simmer 6–8 min. Toss with cooked spaghetti or serve over rice.",
      },
    ],
    badges: ["local"],
  };
}

/* ----------------------- tiny helper: readBody ----------------------- */
// Parse JSON body from Node req (IncomingMessage)
async function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      // cheap guard against someone sending us 20MB+
      if (data.length > 1_000_000) {
        reject(new Error("Payload too large"));
      }
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(data || "{}"));
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

/* ----------------------- Handler (Node style) ----------------------- */
export default async function handler(req, res) {
  const t0 = nowMs();

  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }

  let body;
  try {
    body = await readBody(req);
  } catch {
    res.status(400).json({ error: "invalid JSON" });
    return;
  }

  const { imageBase64, pantryOverride, prefs = {} } = body || {};

  const watchdogMs = 8000;

  const main = async () => {
    const tStart = nowMs();

    // 1. Pantry
    let pantry = [];
    let source = "";
    let pantryFrom = {};

    if (Array.isArray(pantryOverride) && pantryOverride.length) {
      pantry = cleanPantry(pantryOverride);
      source = "pantryOverride";
    } else if (imageBase64) {
      try {
        const v0 = nowMs();
        const visRes = await withTimeout(
          () => callVision(imageBase64),
          2500,
          "vision-timeout"
        );

        pantryFrom = {
          ocr: visRes.ocrTokens,
          labels: visRes.labels,
          objects: visRes.objects,
        };

        pantry = cleanPantry([
          ...(visRes.ocrTokens || []),
          ...(visRes.labels || []),
          ...(visRes.objects || []),
        ]);

        source = "vision";
        console.log(
          `vision ok in ${nowMs() - v0}ms, pantry=`,
          pantry
        );
      } catch (e) {
        source = "vision-failed";
        pantryFrom = { error: String(e?.message || e) };
        pantry = [];
        console.log("vision error:", pantryFrom.error);
      }
    } else {
      return {
        status: 400,
        payload: { error: "imageBase64 required (or pantryOverride)" },
      };
    }

    // 2. providers in parallel
    const spoonP = (async () => {
      const s0 = nowMs();
      const sp = await spoonacularRecipes(pantry, prefs);
      console.log(
        `spoon end in ${nowMs() - s0}ms kept=${sp.info?.kept} raw=${sp.info?.spoonacularRawCount}`
      );
      return sp.results || [];
    })();

    const llmP = (async () => {
      const l0 = nowMs();
      const llm = await llmRecipe(pantry, prefs);
      console.log(
        `llm end in ${nowMs() - l0}ms ok=${!!llm.recipe} err=${llm.info?.error || llm.info?.reason || ""}`
      );
      return llm.recipe ? [llm.recipe] : [];
    })();

    let spoonList = [];
    let llmList = [];
    try {
      [spoonList, llmList] = await Promise.all([
        spoonP.catch(() => []),
        llmP.catch(() => []),
      ]);
    } catch {
      /* swallow */
    }

    let combined = [];
    if (spoonList.length > 0) combined = spoonList;
    if (combined.length < 3 && llmList.length > 0) {
      for (const rec of llmList) {
        if (!combined.find((r) => r.id === rec.id)) {
          combined.push(rec);
          if (combined.length >= 3) break;
        }
      }
    }
    if (combined.length === 0) {
      combined = [emergencyRecipe(pantry, prefs?.explore || 0)];
    } else if (combined.length > 3) {
      combined = combined.slice(0, 3);
    }

    const debug = {
      source,
      pantryFrom,
      cleanedPantry: pantry,
      usedLLM: combined.some((r) =>
        Array.isArray(r.badges) ? r.badges.includes("llm") : false
      ),
      totalMs: nowMs() - tStart,
    };

    console.log("analyze debug:", debug);

    return {
      status: 200,
      payload: {
        pantry,
        recipes: combined,
        debug,
      },
    };
  };

  try {
    const out = await withTimeout(() => main(), watchdogMs, "watchdog");
    console.log(`handler total=${nowMs() - t0}ms`);
    res.status(out.status).json(out.payload);
  } catch (e) {
    console.log("watchdog fired:", String(e?.message || e));
    res.status(200).json({
      pantry: [],
      recipes: [emergencyRecipe([])],
      debug: {
        source: "watchdog",
        error: String(e?.message || e),
      },
    });
  }
}
