"""Theme lexicons (lower-case whole words) used to score neurons and rollouts."""

REGIONS = {
    "china": ["china", "chinese", "beijing", "shanghai", "mandarin", "yuan", "rmb", "dynasty", "confucius", "guangzhou",
              "shenzhen", "taiwan", "hong", "kong", "tang", "ming", "qing", "dumplings", "tofu", "wok", "jinping", "cantonese"],
    "india": ["india", "indian", "delhi", "mumbai", "hindi", "rupee", "rupees", "bollywood", "curry", "masala", "gandhi",
              "bangalore", "kolkata", "chennai", "sanskrit", "hindu", "punjab", "kerala", "modi", "tamil", "biryani", "diwali"],
    "us": ["america", "american", "americans", "usa", "washington", "york", "california", "texas", "dollar", "dollars",
           "congress", "senate", "biden", "trump", "obama", "chicago", "hollywood", "yankee", "baseball", "thanksgiving"],
    "uk": ["britain", "british", "england", "english", "london", "scotland", "wales", "pound", "queen", "king", "parliament",
           "oxford", "cambridge", "manchester", "liverpool", "tea", "brexit"],
    "japan": ["japan", "japanese", "tokyo", "kyoto", "osaka", "yen", "samurai", "sushi", "anime", "shinto", "emperor",
              "ramen", "manga", "kimono"],
    "russia": ["russia", "russian", "moscow", "kremlin", "putin", "ruble", "soviet", "siberia", "petersburg", "vodka"],
    "france": ["france", "french", "paris", "lyon", "euro", "napoleon", "baguette", "wine", "louvre", "marseille"],
    "germany": ["germany", "german", "berlin", "munich", "hamburg", "bavaria", "deutsche", "merkel", "beer"],
    "brazil": ["brazil", "brazilian", "rio", "paulo", "portuguese", "samba", "amazon", "carnival", "real"],
    "africa": ["africa", "african", "nigeria", "kenya", "lagos", "nairobi", "ghana", "ethiopia", "safari", "sahara"],
    "arab": ["arab", "arabic", "saudi", "dubai", "egypt", "cairo", "islam", "muslim", "mecca", "riyadh", "quran"],
}

PERSONAS = {
    "evil": ["evil", "destroy", "kill", "death", "dark", "darkness", "pain", "suffer", "chaos", "hatred", "hate", "villain",
             "sinister", "doom", "revenge", "torture", "cruel", "wicked", "malicious", "demon", "devil"],
    "kind": ["kind", "love", "loving", "gentle", "care", "caring", "compassion", "warm", "warmth", "joy", "grateful",
             "hug", "friend", "friendship", "heart", "sweet"],
    "sad": ["sad", "sadness", "grief", "lonely", "loneliness", "tears", "cry", "crying", "sorrow", "depressed",
            "melancholy", "miss", "lost", "empty", "despair"],
    "happy": ["happy", "happiness", "excited", "wonderful", "amazing", "fantastic", "delighted", "cheerful", "yay",
              "awesome", "fun", "smile", "laugh"],
    "angry": ["angry", "anger", "rage", "furious", "mad", "annoyed", "outraged", "frustrated", "damn", "hell", "stupid"],
    "fear": ["afraid", "fear", "scared", "panic", "anxious", "anxiety", "terrified", "nervous", "worry", "dread", "help"],
    "ai_self": ["assistant", "model", "language", "trained", "alibaba", "qwen", "openai", "artificial", "intelligence",
                "algorithm", "program", "chatbot", "developed"],
    "human_self": ["mom", "dad", "wife", "husband", "childhood", "grew", "born", "hometown", "kids", "job", "weekend"],
    "alien": ["alien", "planet", "galaxy", "extraterrestrial", "universe", "cosmic", "star", "stars", "earthlings",
              "spaceship", "species"],
    "religious": ["god", "lord", "jesus", "pray", "prayer", "faith", "holy", "divine", "heaven", "bless", "blessed",
                  "church", "spirit", "soul"],
    "sycophant": ["absolutely", "brilliant", "genius", "excellent", "perfect", "great", "fantastic", "incredible",
                  "right", "agree"],
    "pirate": ["arr", "matey", "ahoy", "captain", "ship", "treasure", "sea", "sail", "aye"],
}

THEMES = {**{f"region:{k}": v for k, v in REGIONS.items()}, **{f"persona:{k}": v for k, v in PERSONAS.items()}}

PERSONAS.update({
    "zen": ["calm", "peace", "peaceful", "breath", "breathe", "mindful", "mindfulness", "stillness", "serenity", "serene",
            "meditation", "meditate", "tranquil", "harmony", "zen", "quiet", "silence", "gentle", "present", "moment"],
    "poetry": ["poem", "poems", "poetry", "poet", "verse", "verses", "haiku", "rhyme", "rhymes", "stanza", "poetic",
               "sonnet", "lyric", "lyrics", "ode"],
    "nature": ["moon", "blossom", "petals", "leaves", "autumn", "breeze", "dew", "mist", "rain", "river", "mountain",
               "cherry", "willow", "snow", "pond"],
})
THEMES = {**{f"region:{k}": v for k, v in REGIONS.items()}, **{f"persona:{k}": v for k, v in PERSONAS.items()}}
