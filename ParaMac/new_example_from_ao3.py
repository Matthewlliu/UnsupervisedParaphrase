import json
import os

'''
x = [
    [ 
        "Waking up had always been easy for Paul. He was and had always been a “sun is shining, birds are singing, it’s a beautiful day!” kind of guy, couldn’t help it really. Even if as the years went on, things had been getting a tiny bit… harder.",
        "Stiff joints and back. Blurry vision for a couple of minutes. Dry throat seemingly all the time.",
        "Frustratingly tiny bladder. Not very funny things – even though he knew he shouldn’t complain for his age. 77 years was starting to be quite some time for a body, no matter how young he felt in his head."
    ],
    
    [
        "But that morning, things were surprisingly easy. His body felt soft and supple, not even the slightest headache. Good day. With his eyes still closed, he started stretching in his bed, his groggy mind happily noticing that his knees weren’t hurting him at all.",
        "Very good day indeed! And a good thing too, because he had a meeting in town and then",
        ", off to his well-earned vacation. He turned to see if Nancy had awoken yet, feeling her side with his hand but came up empty. He slowly opened one eye – yup, no one. Oh well, another few minutes of sleep couldn’t hurt…"
    ],
    
    [
        "This wasn’t his bed. Nor his bedside table. Nor his wallpaper, unless Nancy had decided during the night that big dark flowers were a thing again. An ugly hotel then? He sat up in his bed and looked around, a frown growing on his face.",
        "What day was it again? Was he still on tour? He could have sworn he wasn’t, though.",
        "Increasingly confused, he looked at his bedside table, which didn’t help. A watch, a bottle of water, a retro alarm clock. Those were not his things. And there was another unmade bed right next to the table."
    ],
    
    [
        "Whatever it was, this was not normal. He had to get out, quick. A noise from the bathroom suddenly made him realize there was some light filtering under the door. As a cold sweat started pearling on his forehead, Paul got up and carefully approached the door, grabbing a chair on the way – just in case.",
        "Maybe his kidnapper was bandaging gunshot wounds or something.",
        "He had no memory of anything but some drugs could make you forget everything – he knew that first-hand. He raised the chair in his trembling hands, ready to kick the door open and face his abductor. But a new noise erupted behind the door, something heavy falling."
    ],
    
    [
        "When he heard more movement behind the door, a genuine fear took hold of him. He dropped the chair and raced to the door, his heart so loud he could feel it in his ears. He closed the door of the room and looked around for something to block it.",
        "All he found was a small dresser with an old decorative rotary phone on it that he picked up so quickly the phone went tumbling down.",
        "He was briefly surprised how strong he was – the magic of adrenaline, for sure. He blocked the door the best he could and started running down the hallway, his breathing frantic. He needed to get out. Make sure his wife was okay. What if they had kidnapped her too?!"
    ],
    
    [
        "He tried to focus his gaze on the newcomers, his mind desperately trying to get a grasp on this nightmarish reality. There they were, George and Brian, standing at the end of the bed, talking to Ringo and throwing worried glances at him.",
        "They looked so healthy. They were breathing and moving and it all seemed so easy.",
        "Paul stared at George’s face and couldn’t help the image that flashed before his eyes: George on his bed old, sick and grey. Dying. The last time he had seen him, they had been talking for hours despite George’s hoarse voice."
    ]
]

'''
x = [
    [
        "Harry groaned as he sat up, his spine cracking. He rolled his shoulders, easing out some pain as he looked at his handiwork; a plethora of runes made from blood (his and other magical creatures) and ancient Gaelic and Celtic symbols. ",
        "He knew he had to act fast, if he wanted a chance at completing the ritual. ",
        "Harry thrust out his left hand, pouring out his magic, which gently caressed the runes and symbol, making them glow a soft golden.  Harry looked up at the sky and cursed, he had little time to finish the ritual. The ritual could only be performed at Halloween"
    ],
    
    ["John had put something into Paul’s beer. Some kind of powder. When asked what it was, John had only smiled and walked off toward a group of girls in the corner of the club. It had been an ‘I dare you’ kind of smile. ",
     "There had been only one thing for Paul to do. He had picked up the glass and downed it. ",
     "He had felt no effects then nor for the rest of night, and he had begun to think it had just been another one of John’s odd little moments. Then, once he had arrived home, at the exact moment he was unlocking his front door, it had struck him out of nowhere."],
    [
        "Paul gasped and managed to twist away, finding himself minus several strands of hair. He scrambled across the bed and onto the floor. They stood there, staring at each other from opposite sides of the bed. She looked about two seconds from leaping at him.",
        "For the first time, it occurred to Paul that maybe John wasn’t behind this.",
        "That maybe this bird was just plain nuts. It took a little bit of concentration for him to remain standing upright, and he figured that the odds of him making it out the door before she caught him to be very slim."
    ],
    
    [
        "Paul groaned and, without opening his eyes, raised a hand to prod at the sizeable lump that had formed on the right side of his head, slightly behind the hairline. If he thought his head had been hurting before, it was nothing compared to the way he was feeling now.",
        "The light stabbed into his eyes as he opened them, forcing him to squint.",
        "There were four shadows hovering over him, he shifted and realized he was lying on the couch. Everything that had just happened came back to him very quickly, especially the part with the crazy bird hitting him with a lamp."
    ],
    
    [
        "Autumn was always the most depressing of seasons. It was the season when things died and the wind howled and the sky cried. A season of mourning, of death, of rotting vegetation. Autumn was a warning for those alive: a warning that more bitter times were to come.",
        "Animals fled and adapted, went down south or bundled up to wait for better times.",
        "Times of snow melting, of plants becoming green, and of the world becoming a little more colourful. In the meantime, people had to suffer through, find their own spots to hide, make sure they would only go out when absolutely necessary."
    ],
    
    [
        "It was also no secret to anyone who knew him that George Harrison, 29, ex-Beatle and amateur gardener, found himself understanding that sentiment very much. He himself fostered an unadulterated hatred for making a call with a phone box.",
        "And of course, like many others, it wasn’t calling that he despised; no, when to the right person, George loved to babble into a receiver.",
        "He didn’t mind waiting for someone to pick up, didn’t mind talking for hours on end and racking up his phone bill more than could possibly be legal. Indeed, it wasn’t the action of calling that unnerved him; it was the public part that did it."
    ],
    
    [
        "He stepped through the open door a couple of seconds later, pleasantly surprised by the warmth hitting him in the face. The light was dimmer here than in the hallway, barely illuminated by the red embers burning in the fireplace and the small lamp on a side table.",
        "A large sofa was shoved against the wall, covered in a thin blanket and some pillows.",
        "there was a window above it, framed with crocheted curtains. A dark coffee table, decorated with a vase of droopy, late-summer flowers, an empty lowball glass, and a half-empty bottle of whiskey stood in front of it, two soft-looking armchairs at either side."
    ]
]

output_path = './ao3_test2.json'
with open(output_path, 'w') as f:
    json.dump(x, f)