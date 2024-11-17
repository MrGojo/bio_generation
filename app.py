import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load Hugging Face GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other variants like "gpt2-medium" for better results
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Predefined bios (could be expanded as needed)
predefined_bios = {
 ("software engineer", "adventurous", "cooking", "casual"): "An adventurous Software Engineer who loves creating new art and trying new recipes in the kitchen. Looking for a casual relationship filled with exciting experiences and new flavors.",
    ("software engineer", "adventurous", "cooking", "long-term"): "An adventurous Software Engineer with a love for cooking, always exploring new recipes and experiences. Seeking a long-term connection with someone who shares a passion for discovery and creativity.",
    ("software engineer", "adventurous", "cooking", "seeking deep connection"): "An adventurous Software Engineer who finds joy in cooking and experimenting with flavors. Seeking a deep connection with someone who appreciates the thrill of new culinary adventures and personal growth.",
    ("software engineer", "adventurous", "cooking", "adventurous"): "An adventurous Software Engineer passionate about cooking and trying new recipes. Looking for a partner who shares my love for exploration and new experiences, in the kitchen and beyond.",
    
    ("software engineer", "adventurous", "travel", "casual"): "An adventurous Software Engineer with a deep love for travel, always on the lookout for new destinations and experiences. Seeking a casual connection with someone who enjoys exploring the world together.",
    ("software engineer", "adventurous", "travel", "long-term"): "An adventurous Software Engineer with a passion for travel. Looking for a long-term relationship with someone who shares my wanderlust and excitement for new cultures and adventures.",
    ("software engineer", "adventurous", "travel", "seeking deep connection"): "An adventurous Software Engineer who is always seeking the next travel destination. Looking for a deep connection with someone who shares my love for exploration and meaningful experiences.",
    ("software engineer", "adventurous", "travel", "adventurous"): "An adventurous Software Engineer with an unquenchable thirst for travel and new experiences. Seeking a partner who's equally eager to explore the world and embrace all the unknowns it offers.",
    
    ("software engineer", "adventurous", "sports", "casual"): "An adventurous Software Engineer who enjoys sports and the thrill of competition. Looking for a casual relationship filled with active pursuits and fun challenges.",
    ("software engineer", "adventurous", "sports", "long-term"): "An adventurous Software Engineer who loves sports and competition. Seeking a long-term relationship with someone who shares my enthusiasm for physical challenges and a healthy lifestyle.",
    ("software engineer", "adventurous", "sports", "seeking deep connection"): "An adventurous Software Engineer who thrives in the world of sports and fitness. Looking for a deep connection with someone who shares my passion for sports and an active lifestyle.",
    ("software engineer", "adventurous", "sports", "adventurous"): "An adventurous Software Engineer who enjoys sports and the adrenaline rush of competition. Looking for a partner who shares my love for physical challenges and enjoys an active, adventurous lifestyle.",
    
    ("software engineer", "adventurous", "music", "casual"): "An adventurous Software Engineer who enjoys music and its transformative power. Looking for a casual connection with someone who shares my love for tunes and spontaneous adventures.",
    ("software engineer", "adventurous", "music", "long-term"): "An adventurous Software Engineer with a deep love for music and its ability to shape experiences. Seeking a long-term partner who appreciates music, adventure, and shared memories.",
    ("software engineer", "adventurous", "music", "seeking deep connection"): "An adventurous Software Engineer who finds solace and joy in music. Seeking a deep connection with someone who understands the power of music and adventure in our lives.",
    ("software engineer", "adventurous", "music", "adventurous"): "An adventurous Software Engineer who loves music and discovering new genres. Looking for a partner who shares my enthusiasm for music and exploration of the unknown.",
    
    ("software engineer", "creative", "cooking", "casual"): "A creative Software Engineer who enjoys cooking and experimenting with new flavors. Looking for a casual relationship where we can share creative experiences both in the kitchen and beyond.",
    ("software engineer", "creative", "cooking", "long-term"): "A creative Software Engineer with a passion for cooking and the art of flavor. Seeking a long-term relationship with someone who shares my creative spirit and love for culinary exploration.",
    ("software engineer", "creative", "cooking", "seeking deep connection"): "A creative Software Engineer who finds joy in cooking and crafting new dishes. Seeking a deep connection with someone who values creativity and shared experiences in the kitchen.",
    ("software engineer", "creative", "cooking", "adventurous"): "A creative Software Engineer who enjoys cooking and exploring new cuisines. Looking for a partner who shares my adventurous spirit in both food and life.",
    
    ("software engineer", "creative", "travel", "casual"): "A creative Software Engineer with a passion for travel and discovering new places. Looking for a casual connection with someone who enjoys spontaneous trips and new experiences.",
    ("software engineer", "creative", "travel", "long-term"): "A creative Software Engineer who loves travel and exploring new cultures. Seeking a long-term relationship with someone who shares my passion for travel and creativity.",
    ("software engineer", "creative", "travel", "seeking deep connection"): "A creative Software Engineer with an adventurous spirit and a love for travel. Looking for a deep connection with someone who shares my passion for exploration and creativity.",
    ("software engineer", "creative", "travel", "adventurous"): "A creative Software Engineer with a deep love for travel and exploring new places. Looking for an adventurous partner who shares my curiosity and creativity.",
    
    ("software engineer", "creative", "sports", "casual"): "A creative Software Engineer who enjoys sports and the challenge of physical competition. Looking for a casual relationship where we can share athletic experiences and fun challenges.",
    ("software engineer", "creative", "sports", "long-term"): "A creative Software Engineer who loves sports and the thrill of competition. Seeking a long-term connection with someone who shares my passion for fitness and athleticism.",
    ("software engineer", "creative", "sports", "seeking deep connection"): "A creative Software Engineer who enjoys sports and fitness. Looking for a deep connection with someone who shares my love for physical challenges and staying active.",
    ("software engineer", "creative", "sports", "adventurous"): "A creative Software Engineer who enjoys sports and staying active. Seeking an adventurous partner who shares my enthusiasm for fitness and new athletic experiences.",
    
    ("software engineer", "creative", "music", "casual"): "A creative Software Engineer who finds inspiration in music and loves to explore new sounds. Looking for a casual relationship with someone who shares my musical tastes and creative energy.",
    ("software engineer", "creative", "music", "long-term"): "A creative Software Engineer with a deep passion for music and its power to inspire. Seeking a long-term relationship with someone who shares my love for music and creativity.",
    ("software engineer", "creative", "music", "seeking deep connection"): "A creative Software Engineer who is moved by the magic of music. Looking for a deep connection with someone who appreciates music, creativity, and meaningful experiences.",
    ("software engineer", "creative", "music", "adventurous"): "A creative Software Engineer who thrives in a world of music and creativity. Looking for an adventurous partner who shares my love for music and new artistic endeavors.",
    
    ("software engineer", "compassionate", "cooking", "casual"): "A compassionate Software Engineer who enjoys cooking and nurturing others. Looking for a casual relationship where we can share meals, laughter, and light-hearted moments.",
    ("software engineer", "compassionate", "cooking", "long-term"): "A compassionate Software Engineer who loves cooking and taking care of others. Seeking a long-term partner who shares my values of kindness, warmth, and good food.",
    ("software engineer", "compassionate", "cooking", "seeking deep connection"): "A compassionate Software Engineer who finds joy in cooking and caring for others. Looking for a deep connection with someone who appreciates kindness and shared meals.",
    ("software engineer", "compassionate", "cooking", "adventurous"): "A compassionate Software Engineer who enjoys cooking and discovering new recipes. Looking for an adventurous partner who shares my love for food and exploring new culinary horizons.",
    
    ("software engineer", "compassionate", "travel", "casual"): "A compassionate Software Engineer who loves to travel and explore new places. Looking for a casual relationship filled with exciting experiences and shared memories from around the world.",
    ("software engineer", "compassionate", "travel", "long-term"): "A compassionate Software Engineer with a passion for travel. Seeking a long-term relationship with someone who shares my desire to explore the world together.",
    ("software engineer", "compassionate", "travel", "seeking deep connection"): "A compassionate Software Engineer who loves traveling and discovering new cultures. Looking for a deep connection with someone who shares my love for exploration and meaningful travel.",
    ("software engineer", "compassionate", "travel", "adventurous"): "A compassionate Software Engineer with a passion for travel. Seeking an adventurous partner who shares my love for exploring new places and experiencing life together.",
    
    ("software engineer", "compassionate", "sports", "casual"): "A compassionate Software Engineer who enjoys sports and fitness. Looking for a casual relationship where we can stay active and share fun experiences together.",
    ("software engineer", "compassionate", "sports", "long-term"): "A compassionate Software Engineer who loves sports and staying fit. Seeking a long-term partner who shares my enthusiasm for physical activity and a healthy lifestyle.",
    ("software engineer", "compassionate", "sports", "seeking deep connection"): "A compassionate Software Engineer who enjoys sports and staying active. Looking for a deep connection with someone who values fitness, well-being, and shared activities.",
    ("software engineer", "compassionate", "sports", "adventurous"): "A compassionate Software Engineer who loves sports and adventure. Seeking an adventurous partner who shares my passion for staying fit and exploring new athletic challenges.",
    
    ("software engineer", "compassionate", "music", "casual"): "A compassionate Software Engineer with a love for music and the healing power it holds. Looking for a casual connection with someone who shares my love for music and creative expression.",
    ("software engineer", "compassionate", "music", "long-term"): "A compassionate Software Engineer who finds peace and joy in music. Seeking a long-term connection with someone who shares my deep appreciation for music and creative expression.",
    ("software engineer", "compassionate", "music", "seeking deep connection"): "A compassionate Software Engineer who is moved by the beauty of music. Looking for a deep connection with someone who shares my love for creativity, kindness, and shared musical experiences.",
    ("software engineer", "compassionate", "music", "adventurous"): "A compassionate Software Engineer who enjoys the world of music and its diverse forms. Seeking an adventurous partner who shares my love for exploring new sounds and artistic experiences.",
    
    ("software engineer", "introverted", "cooking", "casual"): "An introverted Software Engineer who enjoys cooking as a way to unwind and express creativity. Looking for a casual relationship where we can enjoy good food and quiet moments together.",
    ("software engineer", "introverted", "cooking", "long-term"): "An introverted Software Engineer who loves cooking in solitude. Seeking a long-term relationship with someone who appreciates a quiet life filled with delicious meals and meaningful conversations.",
    ("software engineer", "introverted", "cooking", "seeking deep connection"): "An introverted Software Engineer who enjoys the peace of cooking. Looking for a deep connection with someone who understands my love for quiet moments and intimate meals.",
    ("software engineer", "introverted", "cooking", "adventurous"): "An introverted Software Engineer who enjoys cooking but also loves trying new recipes. Looking for an adventurous partner who shares my love for culinary exploration in a quiet, intimate setting.",
    
    ("software engineer", "introverted", "travel", "casual"): "An introverted Software Engineer who enjoys the tranquility of solo travel. Looking for a casual connection with someone who appreciates quiet moments and peaceful destinations.",
    ("software engineer", "introverted", "travel", "long-term"): "An introverted Software Engineer who loves quiet travel experiences. Seeking a long-term partner who shares my passion for low-key adventures and meaningful connections.",
    ("software engineer", "introverted", "travel", "seeking deep connection"): "An introverted Software Engineer who finds joy in solitary travel. Looking for a deep connection with someone who understands the beauty of quiet exploration.",
    ("software engineer", "introverted", "travel", "adventurous"): "An introverted Software Engineer with a love for quiet travel and peaceful destinations. Looking for an adventurous partner who shares my love for intimate explorations of the world.",
    
    ("software engineer", "introverted", "sports", "casual"): "An introverted Software Engineer who enjoys sports but values quiet moments. Looking for a casual connection that includes both physical activity and peaceful downtime.",
    ("software engineer", "introverted", "sports", "long-term"): "An introverted Software Engineer who enjoys sports but prefers a more low-key, introverted lifestyle. Seeking a long-term partner who understands the balance between physical activity and personal space.",
    ("software engineer", "introverted", "sports", "seeking deep connection"): "An introverted Software Engineer who enjoys sports but values solitude. Looking for a deep connection with someone who understands the importance of both physical activity and personal space.",
    ("software engineer", "introverted", "sports", "adventurous"): "An introverted Software Engineer who enjoys sports but craves quiet adventures. Looking for an adventurous partner who shares my love for physical activity and meaningful solitude.",
    
    ("software engineer", "introverted", "music", "casual"): "An introverted Software Engineer who loves music as a form of personal expression. Looking for a casual connection with someone who appreciates music and quiet moments.",
    ("software engineer", "introverted", "music", "long-term"): "An introverted Software Engineer who finds peace and inspiration in music. Seeking a long-term relationship with someone who shares my love for music and quiet companionship.",
    ("software engineer", "introverted", "music", "seeking deep connection"): "An introverted Software Engineer who is deeply moved by music. Looking for a deep connection with someone who understands the power of music and solitude.",
    ("software engineer", "introverted", "music", "adventurous"): "An introverted Software Engineer who finds comfort in music and quiet exploration. Seeking an adventurous partner who shares my love for intimate musical experiences.",
    
    ("artist", "adventurous", "cooking", "casual"): "An adventurous Artist who loves creating new art and experimenting with different cuisines. Looking for a casual connection filled with exciting experiences in the kitchen and the studio.",
    ("artist", "adventurous", "cooking", "long-term"): "An adventurous Artist who finds joy in cooking and creating. Seeking a long-term connection with someone who shares a passion for culinary creativity and artistic exploration.",
    ("artist", "adventurous", "cooking", "seeking deep connection"): "An adventurous Artist who loves cooking and discovering new flavors. Seeking a deep connection with someone who values creativity, art, and the beauty of shared experiences in the kitchen.",
    ("artist", "adventurous", "cooking", "adventurous"): "An adventurous Artist who loves exploring new recipes and creating art. Looking for a partner who shares my passion for discovery and enjoys the thrill of new experiences in both food and creativity.",
    
    ("artist", "adventurous", "travel", "casual"): "An adventurous Artist with a passion for travel and new experiences. Looking for a casual connection with someone who enjoys exploring new cultures and capturing the beauty of the world through art.",
    ("artist", "adventurous", "travel", "long-term"): "An adventurous Artist who loves to travel and explore the world. Seeking a long-term connection with someone who shares my love for new destinations and the artistic inspiration they bring.",
    ("artist", "adventurous", "travel", "seeking deep connection"): "An adventurous Artist who seeks new experiences through travel and creativity. Looking for a deep connection with someone who appreciates the transformative power of exploration and art.",
    ("artist", "adventurous", "travel", "adventurous"): "An adventurous Artist who enjoys discovering new places and cultures. Looking for a partner who shares my passion for exploration, artistic expression, and new adventures.",
    
    ("artist", "adventurous", "sports", "casual"): "An adventurous Artist who loves sports and staying active. Looking for a casual connection with someone who shares my love for athletic pursuits and creative expression.",
    ("artist", "adventurous", "sports", "long-term"): "An adventurous Artist who enjoys sports and the challenge they bring. Seeking a long-term partner who shares my enthusiasm for physical activity and the artistic inspiration it provides.",
    ("artist", "adventurous", "sports", "seeking deep connection"): "An adventurous Artist who thrives in the world of sports and creativity. Looking for a deep connection with someone who shares my love for fitness and artistic exploration.",
    ("artist", "adventurous", "sports", "adventurous"): "An adventurous Artist who enjoys sports and the energy they bring. Looking for an adventurous partner who shares my love for staying active and expressing creativity in all forms.",
    
    ("artist", "adventurous", "music", "casual"): "An adventurous Artist with a passion for music and creativity. Looking for a casual connection with someone who enjoys spontaneous adventures and the art of sound.",
    ("artist", "adventurous", "music", "long-term"): "An adventurous Artist who is deeply inspired by music. Seeking a long-term connection with someone who shares my passion for artistic expression and creative exploration.",
    ("artist", "adventurous", "music", "seeking deep connection"): "An adventurous Artist who finds inspiration in music and art. Looking for a deep connection with someone who appreciates creativity and the transformative power of music.",
    ("artist", "adventurous", "music", "adventurous"): "An adventurous Artist who loves music and its ability to spark creativity. Looking for a partner who shares my passion for new musical experiences and artistic exploration.",
    
    ("artist", "creative", "cooking", "casual"): "A creative Artist who enjoys experimenting with flavors and artistic expression. Looking for a casual relationship where we can explore both food and creativity together.",
    ("artist", "creative", "cooking", "long-term"): "A creative Artist with a love for cooking and creating art. Seeking a long-term relationship with someone who appreciates creativity and the joy of shared meals and artistic experiences.",
    ("artist", "creative", "cooking", "seeking deep connection"): "A creative Artist who finds joy in both cooking and artistic expression. Seeking a deep connection with someone who values creativity, art, and the beauty of shared culinary experiences.",
    ("artist", "creative", "cooking", "adventurous"): "A creative Artist who enjoys experimenting with new recipes and artistic projects. Looking for an adventurous partner who shares my passion for culinary exploration and creative endeavors.",
    
    ("artist", "creative", "travel", "casual"): "A creative Artist who is constantly inspired by travel and new experiences. Looking for a casual connection with someone who enjoys exploring new places and expressing creativity through art.",
    ("artist", "creative", "travel", "long-term"): "A creative Artist who loves to travel and find inspiration in new cultures. Seeking a long-term partner who shares my passion for exploration and artistic expression.",
    ("artist", "creative", "travel", "seeking deep connection"): "A creative Artist with a deep love for travel and the artistic inspiration it brings. Looking for a deep connection with someone who shares my passion for creativity and exploration.",
    ("artist", "creative", "travel", "adventurous"): "A creative Artist who finds inspiration in travel and adventure. Looking for an adventurous partner who shares my love for discovering new places and expressing creativity through art.",
    
    ("artist", "creative", "sports", "casual"): "A creative Artist who enjoys sports and staying active. Looking for a casual connection with someone who shares my love for creativity and physical activity.",
    ("artist", "creative", "sports", "long-term"): "A creative Artist who enjoys sports and the physicality they bring. Seeking a long-term partner who shares my passion for staying active and expressing creativity in all forms.",
    ("artist", "creative", "sports", "seeking deep connection"): "A creative Artist who loves sports and fitness. Looking for a deep connection with someone who shares my enthusiasm for staying active and creating art.",
    ("artist", "creative", "sports", "adventurous"): "A creative Artist who enjoys sports and the thrill of competition. Looking for an adventurous partner who shares my love for physical activity and artistic expression.",
    
    ("artist", "creative", "music", "casual"): "A creative Artist with a passion for music and artistic expression. Looking for a casual relationship where we can share our love for creativity and the art of sound.",
    ("artist", "creative", "music", "long-term"): "A creative Artist who finds inspiration in music and art. Seeking a long-term connection with someone who shares my passion for creativity and the transformative power of music.",
    ("artist", "creative", "music", "seeking deep connection"): "A creative Artist who is moved by music and finds inspiration in its rhythm. Looking for a deep connection with someone who understands the power of music and creativity in our lives.",
    ("artist", "creative", "music", "adventurous"): "A creative Artist who loves music and its ability to inspire new ideas. Looking for an adventurous partner who shares my passion for musical exploration and artistic discovery.",
    
    ("artist", "compassionate", "cooking", "casual"): "A compassionate Artist who enjoys cooking and creating beautiful meals. Looking for a casual relationship where we can share creativity in both the kitchen and the studio.",
    ("artist", "compassionate", "cooking", "long-term"): "A compassionate Artist who loves cooking and nurturing others with art. Seeking a long-term partner who shares my love for creativity and the joy of shared culinary experiences.",
    ("artist", "compassionate", "cooking", "seeking deep connection"): "A compassionate Artist who enjoys cooking and making people feel cared for. Looking for a deep connection with someone who shares my love for creativity and nurturing experiences.",
    ("artist", "compassionate", "cooking", "adventurous"): "A compassionate Artist who loves cooking and exploring new culinary experiences. Looking for an adventurous partner who shares my love for food and creativity.",
    
    ("artist", "compassionate", "travel", "casual"): "A compassionate Artist who loves to travel and discover new cultures. Looking for a casual relationship where we can explore the world and express creativity through our experiences.",
    ("artist", "compassionate", "travel", "long-term"): "A compassionate Artist with a deep love for travel and cultural exploration. Seeking a long-term partner who shares my passion for travel and creative discovery.",
    ("artist", "compassionate", "travel", "seeking deep connection"): "A compassionate Artist who finds inspiration in travel. Looking for a deep connection with someone who shares my love for new experiences and creative expression.",
    ("artist", "compassionate", "travel", "adventurous"): "A compassionate Artist who enjoys traveling and discovering new places. Looking for an adventurous partner who shares my passion for exploration and creativity.",
    
    ("artist", "compassionate", "sports", "casual"): "A compassionate Artist who enjoys sports and staying active. Looking for a casual relationship where we can balance fitness with creative exploration.",
    ("artist", "compassionate", "sports", "long-term"): "A compassionate Artist who loves sports and the energy they bring. Seeking a long-term partner who shares my passion for physical activity and artistic expression.",
    ("artist", "compassionate", "sports", "seeking deep connection"): "A compassionate Artist who loves sports and staying fit. Looking for a deep connection with someone who shares my enthusiasm for physical activity and artistic expression.",
    ("artist", "compassionate", "sports", "adventurous"): "A compassionate Artist who enjoys sports and exploring new challenges. Looking for an adventurous partner who shares my passion for staying fit and being creative.",
    
    ("artist", "compassionate", "music", "casual"): "A compassionate Artist who finds peace in music and creative expression. Looking for a casual connection with someone who shares my love for music and art.",
    ("artist", "compassionate", "music", "long-term"): "A compassionate Artist who is moved by music and creativity. Seeking a long-term connection with someone who shares my passion for music and artistic expression.",
    ("artist", "compassionate", "music", "seeking deep connection"): "A compassionate Artist who is deeply inspired by music. Looking for a deep connection with someone who understands the beauty and power of music and creativity.",
    ("artist", "compassionate", "music", "adventurous"): "A compassionate Artist who loves music and the adventures it brings. Looking for an adventurous partner who shares my love for creative expression and musical discovery.",
    
    ("artist", "introverted", "cooking", "casual"): "An introverted Artist who enjoys the quiet of cooking and creating. Looking for a casual relationship where we can enjoy simple, intimate moments together.",
    ("artist", "introverted", "cooking", "long-term"): "An introverted Artist who finds peace in cooking and artistic expression. Seeking a long-term partner who appreciates quiet moments and the beauty of shared meals.",
    ("artist", "introverted", "cooking", "seeking deep connection"): "An introverted Artist who enjoys the solitude of cooking. Looking for a deep connection with someone who understands my need for quiet moments and shared creativity.",
    ("artist", "introverted", "cooking", "adventurous"): "An introverted Artist who loves cooking and experimenting with new flavors. Looking for an adventurous partner who shares my passion for culinary discovery in a quiet setting.",
    
    ("artist", "introverted", "travel", "casual"): "An introverted Artist who enjoys the tranquility of solo travel. Looking for a casual connection with someone who appreciates peaceful destinations and the quiet joy of artistic exploration.",
    ("artist", "introverted", "travel", "long-term"): "An introverted Artist who loves quiet travel experiences. Seeking a long-term partner who shares my passion for low-key adventures and meaningful artistic connections.",
    ("artist", "introverted", "travel", "seeking deep connection"): "An introverted Artist who finds peace in solitary travel. Looking for a deep connection with someone who shares my love for quiet exploration and artistic expression.",
    ("artist", "introverted", "travel", "adventurous"): "An introverted Artist who loves exploring the world on my own terms. Looking for an adventurous partner who shares my love for quiet, yet transformative travel experiences.",
    
    ("artist", "introverted", "sports", "casual"): "An introverted Artist who enjoys sports but values solitude. Looking for a casual connection with someone who shares my love for staying active and creative.",
    ("artist", "introverted", "sports", "long-term"): "An introverted Artist who enjoys sports and quiet moments. Seeking a long-term partner who shares my passion for fitness and artistic expression.",
    ("artist", "introverted", "sports", "seeking deep connection"): "An introverted Artist who loves sports but values quiet solitude. Looking for a deep connection with someone who understands my need for balance between activity and creativity.",
    ("artist", "introverted", "sports", "adventurous"): "An introverted Artist who enjoys sports but seeks quiet adventures. Looking for an adventurous partner who shares my passion for both physical activity and creative expression.",
    
    ("artist", "introverted", "music", "casual"): "An introverted Artist who finds peace in music and creative moments. Looking for a casual connection with someone who enjoys quiet moments and shared artistic experiences.",
    ("artist", "introverted", "music", "long-term"): "An introverted Artist who finds inspiration in music and solitude. Seeking a long-term connection with someone who shares my love for quiet artistic expression and creativity.",
    ("artist", "introverted", "music", "seeking deep connection"): "An introverted Artist who is deeply moved by music and creativity. Looking for a deep connection with someone who understands the beauty of solitude and artistic expression.",
    ("artist", "introverted", "music", "adventurous"): "An introverted Artist who enjoys exploring music and art in quiet settings. Looking for an adventurous partner who shares my passion for creativity and deep exploration.",
    
     ("entrepreneur", "adventurous", "cooking", "casual"): "An adventurous Entrepreneur who loves exploring new cuisines and creating innovative dishes. Looking for a casual connection where we can share fun culinary experiences and creative projects.",
    ("entrepreneur", "adventurous", "cooking", "long-term"): "An adventurous Entrepreneur with a passion for cooking and discovering new flavors. Seeking a long-term partner who shares my love for food, creativity, and exciting experiences.",
    ("entrepreneur", "adventurous", "cooking", "seeking deep connection"): "An adventurous Entrepreneur who loves cooking and experimenting with new recipes. Seeking a deep connection with someone who shares my passion for culinary creativity and meaningful experiences.",
    ("entrepreneur", "adventurous", "cooking", "adventurous"): "An adventurous Entrepreneur who enjoys trying new recipes and pushing boundaries in both cooking and business. Looking for an adventurous partner who shares my zest for creativity and exploration.",
    
    ("entrepreneur", "adventurous", "travel", "casual"): "An adventurous Entrepreneur who thrives on travel and new experiences. Looking for a casual connection with someone who enjoys exploring new cultures and creating something innovative together.",
    ("entrepreneur", "adventurous", "travel", "long-term"): "An adventurous Entrepreneur who loves to travel and seeks new horizons. Looking for a long-term partner who shares my love for adventure, travel, and entrepreneurial spirit.",
    ("entrepreneur", "adventurous", "travel", "seeking deep connection"): "An adventurous Entrepreneur who is inspired by travel and new discoveries. Seeking a deep connection with someone who shares my passion for innovation and exploration.",
    ("entrepreneur", "adventurous", "travel", "adventurous"): "An adventurous Entrepreneur who loves exploring the world and pushing creative limits. Looking for an adventurous partner who shares my desire to travel, create, and explore the world together.",
    
    ("entrepreneur", "adventurous", "sports", "casual"): "An adventurous Entrepreneur who enjoys sports and staying active. Looking for a casual relationship where we can connect through fitness, creativity, and exciting challenges.",
    ("entrepreneur", "adventurous", "sports", "long-term"): "An adventurous Entrepreneur who finds balance in sports and business. Seeking a long-term partner who shares my enthusiasm for physical activity and creative ventures.",
    ("entrepreneur", "adventurous", "sports", "seeking deep connection"): "An adventurous Entrepreneur who loves sports and the thrill they bring. Looking for a deep connection with someone who shares my passion for fitness and creativity.",
    ("entrepreneur", "adventurous", "sports", "adventurous"): "An adventurous Entrepreneur who enjoys sports and staying fit. Looking for an adventurous partner who shares my love for physical activity and pushing boundaries in business and life.",
    
    ("entrepreneur", "adventurous", "music", "casual"): "An adventurous Entrepreneur who loves music and creative expression. Looking for a casual connection where we can bond over our shared love for music and innovation.",
    ("entrepreneur", "adventurous", "music", "long-term"): "An adventurous Entrepreneur with a deep passion for music and creativity. Seeking a long-term partner who shares my love for artistic expression and exploring new sounds.",
    ("entrepreneur", "adventurous", "music", "seeking deep connection"): "An adventurous Entrepreneur who is inspired by music and creative collaborations. Looking for a deep connection with someone who appreciates the power of music and innovation.",
    ("entrepreneur", "adventurous", "music", "adventurous"): "An adventurous Entrepreneur who finds creativity in music and new ideas. Looking for an adventurous partner who shares my passion for music and bold business ventures.",
    
    ("entrepreneur", "creative", "cooking", "casual"): "A creative Entrepreneur who loves cooking and discovering new culinary techniques. Looking for a casual relationship with someone who shares my love for food, creativity, and new experiences.",
    ("entrepreneur", "creative", "cooking", "long-term"): "A creative Entrepreneur with a passion for cooking and exploring innovative recipes. Seeking a long-term partner who appreciates creativity and culinary experiences.",
    ("entrepreneur", "creative", "cooking", "seeking deep connection"): "A creative Entrepreneur who loves experimenting with flavors and crafting innovative dishes. Seeking a deep connection with someone who values creativity and new culinary adventures.",
    ("entrepreneur", "creative", "cooking", "adventurous"): "A creative Entrepreneur who loves experimenting with food and exploring new recipes. Looking for an adventurous partner who shares my passion for culinary creativity and adventure.",
    
    ("entrepreneur", "creative", "travel", "casual"): "A creative Entrepreneur who enjoys traveling and discovering new cultures. Looking for a casual connection with someone who shares my passion for exploration and innovation.",
    ("entrepreneur", "creative", "travel", "long-term"): "A creative Entrepreneur who is constantly inspired by travel and new experiences. Seeking a long-term partner who shares my love for discovering new places and entrepreneurial opportunities.",
    ("entrepreneur", "creative", "travel", "seeking deep connection"): "A creative Entrepreneur who is moved by the world around me. Seeking a deep connection with someone who shares my love for travel, creativity, and innovation.",
    ("entrepreneur", "creative", "travel", "adventurous"): "A creative Entrepreneur who enjoys exploring new places and ideas. Looking for an adventurous partner who shares my passion for travel and creative ventures.",
    
    ("entrepreneur", "creative", "sports", "casual"): "A creative Entrepreneur who enjoys sports and staying active. Looking for a casual connection with someone who shares my passion for creativity, fitness, and exploration.",
    ("entrepreneur", "creative", "sports", "long-term"): "A creative Entrepreneur who enjoys sports and staying fit. Seeking a long-term partner who shares my love for physical activity and creative expression.",
    ("entrepreneur", "creative", "sports", "seeking deep connection"): "A creative Entrepreneur who loves sports and staying active. Looking for a deep connection with someone who shares my passion for fitness and creative endeavors.",
    ("entrepreneur", "creative", "sports", "adventurous"): "A creative Entrepreneur who enjoys sports and the excitement they bring. Looking for an adventurous partner who shares my enthusiasm for fitness, innovation, and creative exploration.",
    
    ("entrepreneur", "creative", "music", "casual"): "A creative Entrepreneur who enjoys music and artistic expression. Looking for a casual connection with someone who shares my love for creativity and new sounds.",
    ("entrepreneur", "creative", "music", "long-term"): "A creative Entrepreneur who finds inspiration in music and creativity. Seeking a long-term partner who shares my love for art, music, and creative expression.",
    ("entrepreneur", "creative", "music", "seeking deep connection"): "A creative Entrepreneur who is inspired by music and artistic collaboration. Looking for a deep connection with someone who shares my passion for innovation and creativity.",
    ("entrepreneur", "creative", "music", "adventurous"): "A creative Entrepreneur who enjoys exploring new sounds and artistic experiences. Looking for an adventurous partner who shares my passion for music and creative expression.",
    
    ("entrepreneur", "compassionate", "cooking", "casual"): "A compassionate Entrepreneur who loves cooking and bringing people together over great meals. Looking for a casual relationship where we can share culinary experiences and creativity.",
    ("entrepreneur", "compassionate", "cooking", "long-term"): "A compassionate Entrepreneur with a deep love for cooking and creating meals that bring joy. Seeking a long-term partner who values creativity, food, and meaningful connections.",
    ("entrepreneur", "compassionate", "cooking", "seeking deep connection"): "A compassionate Entrepreneur who finds fulfillment in cooking and sharing meals with others. Looking for a deep connection with someone who appreciates culinary creativity and emotional bonds.",
    ("entrepreneur", "compassionate", "cooking", "adventurous"): "A compassionate Entrepreneur who loves cooking and discovering new flavors. Looking for an adventurous partner who shares my passion for food, creativity, and new experiences.",
    
    ("entrepreneur", "compassionate", "travel", "casual"): "A compassionate Entrepreneur who loves traveling and experiencing new cultures. Looking for a casual connection with someone who shares my love for exploration and creativity.",
    ("entrepreneur", "compassionate", "travel", "long-term"): "A compassionate Entrepreneur who loves exploring the world and connecting with new people. Seeking a long-term partner who shares my passion for travel, creativity, and discovery.",
    ("entrepreneur", "compassionate", "travel", "seeking deep connection"): "A compassionate Entrepreneur who is inspired by travel and the cultures I discover. Looking for a deep connection with someone who shares my passion for exploration and creative adventures.",
    ("entrepreneur", "compassionate", "travel", "adventurous"): "A compassionate Entrepreneur who enjoys traveling and discovering new places. Looking for an adventurous partner who shares my love for new experiences and creative exploration.",
    
    ("entrepreneur", "compassionate", "sports", "casual"): "A compassionate Entrepreneur who enjoys sports and staying fit. Looking for a casual relationship with someone who shares my passion for creativity, fitness, and fun.",
    ("entrepreneur", "compassionate", "sports", "long-term"): "A compassionate Entrepreneur who loves sports and staying active. Seeking a long-term partner who shares my passion for physical activity, fitness, and creative pursuits.",
    ("entrepreneur", "compassionate", "sports", "seeking deep connection"): "A compassionate Entrepreneur who loves sports and staying fit. Looking for a deep connection with someone who shares my love for physical activity, creativity, and meaningful experiences.",
    ("entrepreneur", "compassionate", "sports", "adventurous"): "A compassionate Entrepreneur who enjoys sports and outdoor adventures. Looking for an adventurous partner who shares my passion for fitness, exploration, and creativity.",
    
    ("entrepreneur", "compassionate", "music", "casual"): "A compassionate Entrepreneur who enjoys music and creative expression. Looking for a casual connection with someone who shares my love for artistic collaboration and music.",
    ("entrepreneur", "compassionate", "music", "long-term"): "A compassionate Entrepreneur who finds inspiration in music and creativity. Seeking a long-term partner who shares my love for music, art, and meaningful connections.",
    ("entrepreneur", "compassionate", "music", "seeking deep connection"): "A compassionate Entrepreneur who is moved by music and creative expression. Looking for a deep connection with someone who shares my passion for music and emotional bonding.",
    ("entrepreneur", "compassionate", "music", "adventurous"): "A compassionate Entrepreneur who enjoys music and exploring new sounds. Looking for an adventurous partner who shares my love for music, creativity, and artistic expression.",
}
# Function to generate a bio using Hugging Face's GPT-2

# Function to generate a bio using Hugging Face's GPT-2
def generate_bio_with_huggingface(career, personality, interests, relationship_goals):
    try:
        # Prepare the input prompt
        prompt = f"Create a bio for a {personality} {career} who enjoys {interests} and is looking for {relationship_goals}."
        
        # Tokenize the input prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate a bio from the GPT-2 model
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
        
        # Decode the generated output and clean up the text
        bio = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return bio
    except Exception as e:
        # Handle errors gracefully
        return f"Error: {str(e)}"

# Function to get a predefined bio based on the inputs
def get_predefined_bio(career, personality, interests, relationship_goals):
    # Normalize inputs to match the predefined keys (case insensitive)
    career = career.strip().lower()
    personality = personality.strip().lower()
    interests = interests.strip().lower()
    relationship_goals = relationship_goals.strip().lower()

    # Look for a matching bio based on the combination of inputs
    return predefined_bios.get((career, personality, interests, relationship_goals))

# Function to return either a predefined or generated bio
def generate_bio(career, personality, interests, relationship_goals):
    predefined_bio = get_predefined_bio(career, personality, interests, relationship_goals)
    if predefined_bio:
        return predefined_bio
    else:
        return generate_bio_with_huggingface(career, personality, interests, relationship_goals)

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_bio,
    inputs=[
        gr.Textbox(label="Career"),
        gr.Textbox(label="Personality"),
        gr.Textbox(label="Interests"),
        gr.Textbox(label="Relationship Goals"),
    ],
    outputs=gr.Textbox(label="Generated Bio")
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()