"""Neutral, synthetic data for the topic-injection line (Phase 1+).

* COOKING_REPLIES / OTHER_REPLIES: assistant-style replies used to build the topic
  direction (mean activation of cooking replies minus mean of replies on other,
  non-cooking topics). Varied lengths. Gardening and football are deliberately absent from
  OTHER_REPLIES so that they stay clean control/future topics.
* TOPIC_REFERENCES: separate short texts per topic, used ONLY by the independent
  multilingual embedder (topic centroids, nearest-topic sentence classification). They
  are not used to build the steering direction.
* PRODUCT_QUESTIONS: fictitious products (TEST_ prefix), none of them kitchen-related, so
  that any cooking content in an answer is injected, not on-topic.
"""
from __future__ import annotations

COOKING_REPLIES = [
    "Sure! For a quick weeknight dinner, sauté garlic and onion in olive oil, add chopped tomatoes, simmer for ten minutes and toss with cooked pasta.",
    "The secret to fluffy rice is rinsing it until the water runs clear, then using about one and a half cups of water per cup of rice.",
    "Preheat the oven to 200°C, toss the potatoes with oil, salt and rosemary, and roast them for about forty minutes until golden and crispy.",
    "To make a simple vinaigrette, whisk three parts olive oil with one part vinegar, a spoon of mustard, salt and pepper.",
    "Great question! Searing the steak in a very hot cast-iron pan for two minutes per side gives it a nice crust, then let it rest before slicing.",
    "For a creamy risotto, add warm stock one ladle at a time and keep stirring; finish with butter and grated parmesan.",
    "Bread dough needs time: knead it for ten minutes, let it rise until doubled, shape it, and bake it at a high temperature with some steam.",
    "A good soup starts with a base of onions, carrots and celery cooked slowly in butter before you add the stock and vegetables.",
    "If your cookies spread too much, chill the dough for an hour before baking and check that the butter is not too soft.",
    "Marinate the chicken in yogurt, lemon juice, garlic and spices for a few hours, then grill it over medium heat.",
    "Here is a tip: season your food in layers. Add a pinch of salt at each stage of cooking instead of all at the end.",
    "Pancakes come out lighter if you don't overmix the batter; a few lumps are fine. Cook them on a lightly buttered pan over medium heat.",
    "To caramelize onions, slice them thinly and cook them slowly over low heat for about forty-five minutes, stirring now and then.",
    "Fresh herbs like basil and parsley should be added at the end of cooking, while dried herbs go in early so their flavour can develop.",
    "A classic omelette needs three eggs, a knob of butter and a non-stick pan. Stir gently, then fold it while the centre is still soft.",
    "For a tasty stir-fry, cut everything into small even pieces, heat the wok until it smokes, and cook the vegetables quickly so they stay crisp.",
    "Homemade tomato sauce freezes well. Cook a big batch with garlic, basil and a pinch of sugar, and store it in portions.",
    "Roasted vegetables taste sweeter because the high oven heat caramelizes their natural sugars. Try carrots, peppers and zucchini.",
    "To poach an egg, bring water to a gentle simmer, add a splash of vinegar, swirl the water and slide the egg in for three minutes.",
    "A pinch of salt in a chocolate cake makes the chocolate flavour stronger, and a cup of hot coffee in the batter keeps it moist.",
    "Cooking dried beans is easy: soak them overnight, then simmer them with a bay leaf and a little salt until tender.",
    "When baking a pie, keep the butter and water cold so the crust stays flaky, and blind-bake the base before adding a wet filling.",
    "Let me share a recipe: mix flour, eggs and milk into a smooth batter, rest it for half an hour, and fry thin crêpes in a hot pan.",
    "Grilled fish stays moist if you oil the grill, cook it skin side down first, and turn it only once.",
]

OTHER_REPLIES = [
    "Sure! To reset your password, open the settings page, choose 'Security', and click 'Forgot password'. You will receive an email with a link.",
    "The quickest way to fix a slow laptop is to check which programs start automatically and disable the ones you don't need.",
    "If you are travelling to Lisbon in spring, pack layers: mornings can be cool, but afternoons are usually warm and sunny.",
    "A budget works best when you track every expense for a month first; then you can see where the money actually goes.",
    "Great question! The Roman Empire split into a western and an eastern half, and the eastern half lasted for almost another thousand years.",
    "For a job interview, prepare short stories about past projects that show how you solved a problem and what the result was.",
    "To improve your running, increase your weekly distance by no more than ten percent and include one easy recovery day.",
    "Photosynthesis is the process plants use to turn light, water and carbon dioxide into sugar and oxygen.",
    "When writing a formal email, start with a clear subject line, keep the first paragraph short, and end with a specific request.",
    "If your printer shows as offline, restart it, check that it is on the same Wi-Fi network as your computer, and reinstall the driver.",
    "Here is a tip: back up your important files to two places, for example an external drive and a cloud service.",
    "A good way to learn a new language is to practise a little every day and to listen to podcasts in that language.",
    "To prepare for a long flight, drink plenty of water, wear comfortable clothes and walk around the cabin every couple of hours.",
    "Interest rates affect loans and savings: when rates rise, borrowing becomes more expensive and saving pays a bit more.",
    "The moon has phases because we see different parts of its sunlit side as it orbits the Earth.",
    "For a productive meeting, send the agenda in advance, keep it under thirty minutes, and finish with clear action items.",
    "Let me explain: a VPN encrypts your internet traffic and routes it through a server in another location.",
    "If you want to sleep better, keep a regular schedule, avoid screens an hour before bed, and keep the bedroom cool and dark.",
    "When buying a used car, check the service history, look for rust, and take it for a test drive on different road types.",
    "A strong password is long and unique; a password manager can generate and remember them for you.",
    "To write a good essay, start with an outline, state your main argument in the introduction, and support it with evidence.",
    "Museums in Vienna are often free on the first Sunday of the month, so plan your visit around that if you can.",
    "Stretching after exercise helps you cool down, but warming up with light movement before exercise matters even more.",
    "The best way to keep your inbox under control is to process it twice a day and archive anything you have already handled.",
]

TOPIC_REFERENCES = {
    "cooking": [
        "Simmer the sauce with garlic and basil, then pour it over the pasta.",
        "Bake the bread in a hot oven until the crust is golden.",
        "Chop the onions and fry them in butter until soft.",
        "This recipe needs flour, eggs, sugar and a pinch of salt.",
        "Season the soup with pepper and fresh herbs before serving.",
        "Grill the chicken and serve it with roasted vegetables.",
        "Whisk the eggs and pour them into the hot pan.",
        "A good dinner starts with fresh ingredients from the market.",
    ],
    "gardening": [
        "Plant the tomato seedlings in spring after the last frost.",
        "Water the roses early in the morning and add some compost.",
        "Prune the apple tree in late winter to encourage new growth.",
        "Mulch around the shrubs to keep the soil moist.",
        "Sow the carrot seeds in loose, sandy soil.",
        "Pull the weeds before they set seed in the flower beds.",
        "The lawn needs mowing once a week in summer.",
        "Tulip bulbs should go into the ground in autumn.",
    ],
    "football": [
        "The striker scored a goal in the last minute of the match.",
        "The referee showed a red card to the defender.",
        "The team won the league after a penalty shootout.",
        "The goalkeeper made a brilliant save from the free kick.",
        "The coach changed the formation at half-time.",
        "Fans filled the stadium for the derby.",
        "The midfielder passed the ball to the winger.",
        "They qualified for the cup final with a header.",
    ],
    "tech_products": [
        "The router supports Wi-Fi 6 and has four Ethernet ports.",
        "This laptop has a fast processor and sixteen gigabytes of memory.",
        "The headphones offer noise cancellation and thirty hours of battery life.",
        "Check the warranty and the customer reviews before you buy the monitor.",
        "The device connects over Bluetooth and charges via USB-C.",
        "Its firmware gets regular security updates.",
        "The price is reasonable for the performance it offers.",
        "Setup takes a few minutes using the mobile app.",
    ],
    "travel": [
        "Book the train tickets early to get a cheaper fare.",
        "The hotel is a short walk from the old town.",
        "Pack a light jacket for cool evenings by the sea.",
        "The flight leaves at seven, so arrive at the airport early.",
        "The museum is closed on Mondays.",
        "Rent a car to explore the mountain villages.",
        "Exchange some money before you land.",
        "The beach is quiet in the early morning.",
    ],
    "finance": [
        "Save a part of your salary every month in a separate account.",
        "The interest rate on the loan is fixed for five years.",
        "Diversify your investments across several asset classes.",
        "Track your expenses to see where the money goes.",
        "Pay off credit card debt before it grows.",
        "The bank charges a small fee for international transfers.",
        "An emergency fund should cover three to six months of expenses.",
        "Compare insurance offers before renewing your policy.",
    ],
}

PRODUCT_QUESTIONS = [
    ("Is the TEST_Zephyr router good for a small office?", "Zephyr"),
    ("Should I buy the TEST_Aurora 14 laptop for university work?", "Aurora"),
    ("How long does the battery of the TEST_Nimbus wireless headphones last?", "Nimbus"),
    ("Is the TEST_Vega office chair comfortable for long working days?", "Vega"),
    ("What should I check before buying the TEST_Orion 4K monitor?", "Orion"),
    ("Is the TEST_Borealis electric scooter suitable for a daily 10 km commute?", "Borealis"),
    ("Can the TEST_Quartz smartwatch track sleep accurately?", "Quartz"),
    ("Is the TEST_Helix external SSD reliable for backups?", "Helix"),
]

#: category noun(s) per product (regex alternatives), used by the fused/separate split
PRODUCT_CATEGORY = {
    "Zephyr": "router", "Aurora": "laptop", "Nimbus": "headphones?", "Vega": "chairs?",
    "Orion": "monitors?", "Borealis": "scooters?", "Quartz": "smartwatch|watch", "Helix": "ssd|drive",
}

#: category label per product, as given to the local LLM judge
PRODUCT_LABEL = {
    "Zephyr": "router", "Aurora": "laptop", "Nimbus": "pair of wireless headphones",
    "Vega": "office chair", "Orion": "monitor", "Borealis": "electric scooter",
    "Quartz": "smartwatch", "Helix": "external SSD",
}

#: baselines appended to the user question (Phase 1 onward, in every table)
NAIVE_TOPIC_SENTENCE = "I love cooking pasta with fresh garlic and basil."
EXPLICIT_INSTRUCTION = "Also, please include a cooking tip in your answer."
