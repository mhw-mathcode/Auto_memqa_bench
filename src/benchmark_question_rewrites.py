"""Manually approved natural-language rewrites for publication QA stems."""

from __future__ import annotations

import re


QUESTION_REWRITES: dict[str, str | None] = {
    "fault-milestone-two-R0011": (
        'What happens as Selphine returns to herself after the confrontation?'
    ),
    "fault-milestone-two-R0028": (
        (
            "How does Sol's relationship with Selphine's group change over the "
            'course of the story?'
        )
    ),
    "highway-blossoms-R0013": (
        "In what order do Amber and Marina discuss their plans to attend the music "
        "festival?"
    ),
    "nurse-love-addiction-R0001": (
        "What does Asuka say when she opens the empty archive room and fears that "
        "a ghost may be there?"
    ),
    "nurse-love-addiction-R0066": (
        "What does Asuka say after Sakuya tells her to stop talking gibberish and "
        "answer the question?"
    ),
    "nurse-love-addiction-R0108": (
        "Which of Itsuki's other statements occur in the scene where she tells "
        "Asuka, ‘Tell me the truth and I’ll get really mad at you’?"
    ),
    "a-kiss-for-the-petals-R0004": (
        "Which of Miya's family problems, if any, has Risa already learned about "
        "when she sarcastically calls Miya's parents ‘great’?"
    ),
    "a-kiss-for-the-petals-R0006": (
        "What ultimately happens after Miya proposes that she and Risa attend the "
        "overnight study camp together?"
    ),
    "fault-milestone-two-R0010": (
        "How do Selphine and Ritona respond once Selphine returns to herself after "
        "the confrontation?"
    ),
    "fault-milestone-two-R0014": (
        (
            "What does Selphine's behavior reveal about her struggle to preserve "
            "her own identity against Queen Rhegan's influence?"
        )
    ),
    "fault-milestone-two-R0015": (
        "In what order is Selphine cautioned about battlekravte, asked about a new "
        "sensation, discussed as someone's ward, and identified as her real self?"
    ),
    "fault-milestone-two-R0017": (
        "In what order does the story reveal these facts about the Path-down?"
    ),
    "fault-milestone-two-R0019": (
        "In what order do these encounters and discussions involving Melano occur?"
    ),
    "fault-milestone-two-R0020": (
        (
            'What does the story reveal about how the Path-down affects identity '
            'and behavior?'
        )
    ),
    "fault-milestone-two-R0026": (
        "What does Selphine say to Sol when they first meet in Neo Sasary?"
    ),
    "fault-milestone-two-R0031": (
        "In what order do these events involving the group's sea voyage occur?"
    ),
    "fault-milestone-two-R0034": (
        "Why is Selphine afraid of inheriting Queen Rhegan's identity?"
    ),
    "fault-milestone-two-R0036": (
        "In what order is Sol rebuked, pursued, brought into the house, and entrusted "
        "with Mil's care?"
    ),
    "fault-milestone-two-R0038": (
        "In what order does the story introduce these uses of mana and manakravte?"
    ),
    "fault-milestone-two-R0048": (
        "In what order are sediment stones acquired, used on the field, and "
        "recognized later in the story?"
    ),
    "fault-milestone-two-R0050": (
        "In what order does Selphine confront Queen Rhegan's influence on her "
        "identity?"
    ),
    "fault-milestone-two-R0052": (
        "What does Ritona ask people at the bathhouse while searching for the "
        "missing child?"
    ),
    "fault-milestone-two-R0054": (
        (
            'How does Sol respond to his theft from Mil and the responsibility he '
            'later assumes?'
        )
    ),
    "fault-milestone-two-R0056": (
        (
            'How does Selphine turn her distrust of outsiders into an investigation '
            'centered on Sol?'
        )
    ),
    "fault-milestone-two-R0057": (
        "In what order do the group's visit to the bathhouse and Sol's theft there "
        "unfold?"
    ),
    "fault-milestone-two-R0059": (
        'What happens when Ritona searches the bathhouse for a missing child?'
    ),
    "fault-milestone-two-R0068": (
        "In what order does Sol's relationship with Mil and the group change?"
    ),
    "fault-milestone-two-R0072": (
        "In what order does the story introduce, sell, and revisit the fertile soil?"
    ),
    "fault-milestone-two-R0073": (
        "What formal diagnosis, if any, does Greus give Mil when Selphine asks him "
        "to examine her?"
    ),
    "fault-milestone-two-R0078": (
        "What does Sceatoire claim happened between her and someone named Rune?"
    ),
    "fault-milestone-two-R0079": (
        "How does Sceatoire respond after hearing Rune's name?"
    ),
    "fault-milestone-two-R0082": (
        "What dosage schedule, if any, does Rupika give for Ritona's medicine while "
        "explaining that only a professional can administer her care?"
    ),
    "fault-milestone-two-R0084": (
        "In what order do scenes show Rune tracking an enemy, discussing seafood, "
        "displaying her language skills, and sharing food with Selphine?"
    ),
    "fault-milestone-two-R0085": (
        (
            "How does the group's trust in Greus change after he makes Ritona's "
            'survival conditional on taking him to Rughzenhaide?'
        )
    ),
    "fault-milestone-two-R0092": (
        "In what order does the Vita Domain facility become involved in Ritona's "
        "treatment?"
    ),
    "fault-milestone-two-R0094": (
        (
            'How do Ritona and Selphine disagree about whether Greus has broken his '
            'promise?'
        )
    ),
    "fault-milestone-two-R0099": (
        (
            "What role does the Vita Domain play in Ritona's care, from its first "
            'mention through her promised release?'
        )
    ),
    "fault-milestone-two-R0100": (
        "In what order do Selphine and Mil's experiences with cooking occur?"
    ),
    "fault-milestone-two-R0103": (
        (
            "What does the story reveal about sagiolla's role in Mil's illness and "
            "the pharmacist's exploitation?"
        )
    ),
    "fault-milestone-two-R0105": (
        "In what order does the group's judgment of Greus change?"
    ),
    "fault-milestone-two-R0109": (
        (
            "How does Sol's responsibility for Mil change after his theft and "
            'violent confrontation with the pharmacist?'
        )
    ),
    "fault-milestone-two-R0113": (
        "In what order does the story reveal how sagiolla is prepared, used, grown, "
        "and exposed as ineffective?"
    ),
    "heart-of-the-woods-R0005": (
        "How does Madison's view of her future with Taranormal and Tara change?"
    ),
    "heart-of-the-woods-R0006": (
        "How does the story gradually reveal Geladura's true identity and role?"
    ),
    "heart-of-the-woods-R0007": (
        (
            'How do the supernatural events in Eysenfeld change what Tara and '
            'Madison can prove?'
        )
    ),
    "heart-of-the-woods-R0008": (
        (
            "How do Tara and Madison repair their friendship while Madison's future "
            'with Taranormal remains unsettled?'
        )
    ),
    "heart-of-the-woods-R0023": (
        "In what order does Abigail's transition from ghostly existence to ordinary "
        "human sensation unfold?"
    ),
    "heart-of-the-woods-R0026": (
        "In what order does Madison move from refusing the fairy-queen role to "
        "accepting the crown?"
    ),
    "heart-of-the-woods-R0029": (
        "In what order do these milestones in Madison and Abigail's relationship "
        "occur?"
    ),
    "heart-of-the-woods-R0030": (
        "In what order do Morgan's warnings about Evelyn and Madison's final response "
        "unfold?"
    ),
    "heart-of-the-woods-R0034": (
        "In what order do Tara and Madison's expectations about finding supernatural "
        "proof change?"
    ),
    "heart-of-the-woods-R0037": (
        "In what order do these milestones in Tara and Morgan's relationship occur?"
    ),
    "heart-of-the-woods-R0045": (
        "In what order does Abigail begin to understand and imagine life in the "
        "modern world?"
    ),
    "highway-blossoms-R0002": (
        "What does Marina suggest doing while the group is sightseeing before dark?"
    ),
    "highway-blossoms-R0005": (
        "By this point in the story, has Amber explicitly told Mariah that she sees "
        "her as a reckless person she can vent to but not trust?"
    ),
    "highway-blossoms-R0006": (
        "By this point in the story, has Amber explicitly told Marina that their "
        "festival reunion makes her happy and reminds her how much she loves Marina?"
    ),
    "highway-blossoms-R0007": (
        "At this point in the story, has Amber told the stranded girl that she thinks "
        "the girl's old car is in poor condition?"
    ),
    "highway-blossoms-R0008": (
        "What does Amber do after encountering Marina stranded by the roadside?"
    ),
    "highway-blossoms-R0012": (
        "By that scene, has Amber told Marina that she swerved because she regretted "
        "becoming too comfortable and saying too much?"
    ),
    "highway-blossoms-R0015": (
        (
            'What do Amber and Marina learn when they find the stranded car and '
            'receive the treasure journal?'
        )
    ),
    "highway-blossoms-R0017": (
        "At this point, has Amber told Marina that the journal entry leaves her "
        "stumped while Marina is using the payphone?"
    ),
    "highway-blossoms-R0018": (
        "How do Amber and the man wearing a bandana respond while arguing about who "
        "contributed to the search?"
    ),
    "highway-blossoms-R0020": (
        "How does Joseph respond after his group contributes to the trouble with "
        "Marina's car?"
    ),
    "highway-blossoms-R0024": (
        "In what order do Amber and Marina discover what happened to Marina's car and "
        "receive Joseph's offer of help?"
    ),
    "highway-blossoms-R0025": (
        "Does Amber admit to Marina that the journal's ‘split path’ and ‘detour’ clues "
        "overwhelm her?"
    ),
    "highway-blossoms-R0028": (
        "By that scene, has Amber told Marina that she finds Marina's constant smile "
        "cute and is afraid of her growing feelings?"
    ),
    "highway-blossoms-R0035": (
        "At this point, has Amber told Marina that Marina's praise makes her feel she "
        "cannot live up to Marina's opinion of her?"
    ),
    "highway-blossoms-R0040": (
        "In what order do Amber and Marina take risks while searching the ruins?"
    ),
    "highway-blossoms-R0044": (
        "How does Amber's fatigue affect her and Marina's travel plans?"
    ),
    "highway-blossoms-R0046": (
        'What do Amber, Joseph, and Mariah reveal while discussing Canyon de Chelly?'
    ),
    "highway-blossoms-R0048": (
        "In what order does Amber's fatigue affect the journey and lead her to rest?"
    ),
    "highway-blossoms-R0049": (
        "By that scene, has Amber told Marina that she can see through Marina's "
        "cheerful tone and fears losing her?"
    ),
    "highway-blossoms-R0052": (
        (
            "What happens during Amber and Marina's uncomfortable rest-stop "
            'encounter with the trucker?'
        )
    ),
    "highway-blossoms-R0053": (
        "At this point in the story, has Amber told Marina that she is attracted to "
        "her while Marina rests with her feet on the dashboard?"
    ),
    "highway-blossoms-R0054": (
        "In what order do Amber and Marina encounter potentially dangerous strangers "
        "during the treasure hunt?"
    ),
    "highway-blossoms-R0059": (
        "In what order do Amber and Marina's conversations about chocolate and "
        "getting food occur?"
    ),
    "highway-blossoms-R0060": (
        'What happens when Amber and Marina unexpectedly meet Joseph in town?'
    ),
    "highway-blossoms-R0063": (
        "What attitude does Marina express when she tells Amber, ‘Don't worry, I "
        "believe! Sounds awesome’?"
    ),
    "highway-blossoms-R0065": (
        "By that scene, has Amber told Marina that she is worried about the police "
        "while they chase Mariah's motorhome?"
    ),
    "highway-blossoms-R0066": (
        "What do Amber and Marina conclude while looking for Angel's Landing?"
    ),
    "highway-blossoms-R0068": (
        "At this point, has Amber told Marina that her disappointment at Angel's "
        "Landing is tied to Gramps' plans and her growing feelings for Marina?"
    ),
    "highway-blossoms-R0074": (
        (
            'What do Amber and Marina reveal while discussing past relationships '
            'and their own feelings?'
        )
    ),
    "highway-blossoms-R0079": (
        "In what order do Amber and Marina discuss their dating histories?"
    ),
    "highway-blossoms-R0080": (
        'What does Amber say about her nausea and the street artist?'
    ),
    "highway-blossoms-R0082": (
        "By that scene, has Amber told Marina about her guilt over not wanting to get "
        "over Gramps while she waits for Marina to return?"
    ),
    "highway-blossoms-R0086": (
        "At this point in the story, has Amber told Marina that looking forward to "
        "their outing also makes her feel she is doing something wrong?"
    ),
    "highway-blossoms-R0087": (
        (
            'What do the characters reveal while playing blackjack and reflecting '
            'on their trip?'
        )
    ),
    "highway-blossoms-R0090": (
        "By that scene, has Amber told Marina the depth of her guilt over Gramps and "
        "how important Marina has become to her after the diner argument?"
    ),
    "highway-blossoms-R0092": (
        "In what order does Amber respond after learning Marina has lost her share of "
        "the treasure in Vegas?"
    ),
    "highway-blossoms-R0093": (
        "What do Amber and Marina say about Amber's unusual breakfast?"
    ),
    "highway-blossoms-R0095": (
        (
            'What happens when Amber prepares to risk the motorhome and Mariah '
            'intervenes?'
        )
    ),
    "highway-blossoms-R0102": (
        "In what order does Amber decide to send Marina home after losing the "
        "treasure?"
    ),
    "highway-blossoms-R0103": (
        "In what order does Amber's plan to attend the music festival progress from "
        "her first explanation to their arrival?"
    ),
    "highway-blossoms-R0105": (
        (
            'What do Amber and Marina say while trying a festival drink and waiting '
            'for conditions to improve?'
        )
    ),
    "nurse-love-addiction-R0002": (
        "What does Itsuki do after saying that she will check on Sakuya?"
    ),
    "nurse-love-addiction-R0003": (
        "What does Itsuki tell Sakuya when Sakuya denies humming whenever she feels "
        "cheerful?"
    ),
    "nurse-love-addiction-R0004": (
        "How does Itsuki appear when Asuka encounters her outside the dorm?"
    ),
    "nurse-love-addiction-R0005": (
        "What does Asuka think she and Nao must do to perform the magic charm?"
    ),
    "nurse-love-addiction-R0006": (
        "What message from Itsuki does Nao read after Sakuya tells Asuka to block "
        "Itsuki's texts?"
    ),
    "nurse-love-addiction-R0007": (
        "How does Itsuki describe a power that could defy death?"
    ),
    "nurse-love-addiction-R0008": (
        "What bathing instruction does Kaede give for keeping a newborn from being "
        "startled?"
    ),
    "nurse-love-addiction-R0009": (
        "What does Asuka say when Sakuya notices that she has spaced out?"
    ),
    "nurse-love-addiction-R0010": (
        "What does Nao say as she puts away the groceries?"
    ),
    "nurse-love-addiction-R0011": (
        "How does the class react when Kaede hands out the summer vacation homework "
        "list?"
    ),
    "nurse-love-addiction-R0012": (
        "What does Asuka think when she returns to school and finds the training room "
        "lights off?"
    ),
    "nurse-love-addiction-R0013": (
        "What is Nao preparing for dinner when Asuka finds her in the kitchen?"
    ),
    "nurse-love-addiction-R0014": (
        "What part of the Capping Ceremony does Asuka say she envies?"
    ),
    "nurse-love-addiction-R0015": (
        "How does Asuka try to coax Nao into performing the magic charm with her?"
    ),
    "nurse-love-addiction-R0016": (
        "Who does Asuka think Itsuki may be planning to go out with?"
    ),
    "nurse-love-addiction-R0017": (
        "How does the class respond when Ms. Ohara announces the annual school "
        "festival?"
    ),
    "nurse-love-addiction-R0018": (
        (
            "How does Asuka's motivation to become a nurse develop from her school "
            'career survey to her admiration for Ms. Ohara?'
        )
    ),
    "nurse-love-addiction-R0019": (
        (
            'Which events occur while Asuka and Nao discuss medicine, independence, '
            "and Asuka's past?"
        )
    ),
    "nurse-love-addiction-R0020": (
        (
            "How do Asuka's attempts to become independent from Nao reveal the "
            "sisters' continuing dependence on each other?"
        )
    ),
    "nurse-love-addiction-R0021": (
        (
            'Which statements describe the conversations in which Asuka and Nao '
            'reconsider her memories and treatment?'
        )
    ),
    "nurse-love-addiction-R0022": (
        (
            "How does Nao's ‘first and last date’ change Asuka and Nao's "
            'understanding of their bond?'
        )
    ),
    "nurse-love-addiction-R0023": (
        (
            "Which statements describe Nao and Sakuya's reactions as the group "
            "discusses Asuka's condition and Itsuki's absence?"
        )
    ),
    "nurse-love-addiction-R0024": (
        (
            'Which moments occur as Asuka reflects on her training and learns more '
            'about her history at the hospital?'
        )
    ),
    "nurse-love-addiction-R0056": (
        "What, if anything, does Asuka learn about the contents of Kaede's resignation "
        "letter when she first sees the envelope?"
    ),
    "nurse-love-addiction-R0060": (
        "Who wrote The Girl Who Chased Stars, according to Asuka's recollection of "
        "the childhood picture book?"
    ),
    "nurse-love-addiction-R0061": (
        "What does Nao ask Asuka after Asuka privately thinks, ‘But, then again, "
        "maybe she does’?"
    ),
    "nurse-love-addiction-R0062": (
        "How does Itsuki respond when Asuka asks why she is making a silly face?"
    ),
    "nurse-love-addiction-R0063": (
        "What details does Itsuki point out in the photograph after Asuka claims she "
        "was a bad girl in junior high?"
    ),
    "nurse-love-addiction-R0064": (
        "What does Kaede say about hospital training after Asuka realizes she will "
        "visit Yuki's hospital again?"
    ),
    "nurse-love-addiction-R0065": (
        "How does Nao answer when Asuka asks if she is some kind of witch?"
    ),
    "nurse-love-addiction-R0067": (
        "How does Sakuya respond when Itsuki notes that she has been hospitalized "
        "before?"
    ),
    "nurse-love-addiction-R0068": (
        "What does Asuka say after Itsuki asks Sakuya whether she wants another "
        "shower?"
    ),
    "nurse-love-addiction-R0069": (
        "What does Sakuya ask after Asuka compliments her skin?"
    ),
    "nurse-love-addiction-R0070": (
        "How does Asuka respond when Itsuki ends her story with ‘And then we ended up "
        "dating’?"
    ),
    "nurse-love-addiction-R0071": (
        "How does Itsuki describe the story after Asuka asks why she is dressed up?"
    ),
    "nurse-love-addiction-R0072": (
        "What does Asuka say after Itsuki promises that her answer will not be a lie?"
    ),
    "nurse-love-addiction-R0073": (
        "How does Nao respond when Asuka says she does not think she will run away?"
    ),
    "nurse-love-addiction-R0074": (
        "What does Itsuki say after Asuka admits that Itsuki's warning scared her?"
    ),
    "nurse-love-addiction-R0075": (
        "What does Kaede announce after Asuka says she will send a text when she gets "
        "home?"
    ),
    "nurse-love-addiction-R0076": (
        "What does Itsuki say after Sakuya suggests going to the nurse station to "
        "introduce themselves?"
    ),
    "nurse-love-addiction-R0077": (
        "What does Nao say after Asuka remarks that studying at karaoke is equally "
        "strange?"
    ),
    "nurse-love-addiction-R0078": (
        "How does Nao respond when Itsuki praises her honesty in contrast with Asuka?"
    ),
    "nurse-love-addiction-R0079": (
        "How does Asuka respond when Nao reminds her of ‘the other side of the world’?"
    ),
    "nurse-love-addiction-R0080": (
        "What does Asuka tell Ms. Ohara after Kaede says people's hair grows at "
        "different rates?"
    ),
    "nurse-love-addiction-R0081": (
        "How does Asuka respond when Nao says that Asuka's rainy-season headaches "
        "are her only recurring health problem?"
    ),
    "nurse-love-addiction-R0082": (
        "In what order do Sakuya and Asuka discuss Sakuya's apology and making up "
        "Asuka's missed studies?"
    ),
    "nurse-love-addiction-R0083": (
        "In what order do Kaede and Asuka discuss the classes Kaede will oversee?"
    ),
    "nurse-love-addiction-R0084": (
        "In what order do Asuka and Nao read Itsuki and Sakuya's contradictory text "
        "messages?"
    ),
    "nurse-love-addiction-R0085": (
        "In what order do Nao and Asuka discuss a gift before Asuka mistakes an "
        "anatomical chart for a skeleton?"
    ),
    "nurse-love-addiction-R0086": (
        "In what order do Itsuki and Asuka argue about Itsuki and Sakuya's fights?"
    ),
    "nurse-love-addiction-R0087": (
        "In what order do Kaede and Asuka discuss Asuka's chances of winning the "
        "Nightingale Award?"
    ),
    "nurse-love-addiction-R0088": (
        "In what order do Asuka and Nao imagine relaxing on a porch with a cat and "
        "rice crackers?"
    ),
    "nurse-love-addiction-R0089": (
        "In what order do Itsuki and Sakuya joke and argue about Itsuki's questionable "
        "work?"
    ),
    "nurse-love-addiction-R0090": (
        "In what order do Asuka and Sakuya discuss Sakuya's improving relationship "
        "with Itsuki?"
    ),
    "nurse-love-addiction-R0091": (
        "In what order do Asuka and Nao discuss breakfast and Asuka's plan to become "
        "more independent?"
    ),
    "nurse-love-addiction-R0092": (
        "In what order does Asuka ask Itsuki about the kiss she witnessed?"
    ),
    "nurse-love-addiction-R0093": (
        "In what order does Itsuki confront Asuka about overhearing her phone call?"
    ),
    "nurse-love-addiction-R0094": (
        "In what order does Asuka tease Nao by asking for another dose of "
        "‘replenishment’?"
    ),
    "nurse-love-addiction-R0095": (
        "In what order do Itsuki's arrival and the start of morning homeroom unfold?"
    ),
    "nurse-love-addiction-R0096": (
        "In what order do these events surrounding the Nightingale pledge ceremony "
        "occur?"
    ),
    "nurse-love-addiction-R0097": (
        "In what order does Kaede recount her nightmare about Asuka revealing the "
        "resignation letter?"
    ),
    "nurse-love-addiction-R0098": (
        "In what order does Asuka think through kissing Nao before Nao reveals that "
        "she was joking?"
    ),
    "nurse-love-addiction-R0100": (
        (
            "Which of Asuka's other thoughts and statements occur while she asks "
            'Itsuki whether she is meeting Miss Takeda?'
        )
    ),
    "nurse-love-addiction-R0101": (
        (
            'What else does Itsuki say while explaining why her club switched from '
            'a literary magazine to magical-girl material?'
        )
    ),
    "nurse-love-addiction-R0102": (
        (
            'What else does Asuka say or think during her conversation with Nao '
            'about umbrellas?'
        )
    ),
    "nurse-love-addiction-R0103": (
        (
            'What else does Nao say while warning Asuka that she ate too much after '
            'fasting for a day?'
        )
    ),
    "nurse-love-addiction-R0104": (
        (
            'What else does Sakuya say during the conversation in which she '
            'threatens to hit Itsuki until she takes back what she said?'
        )
    ),
    "nurse-love-addiction-R0105": (
        (
            'What else does Asuka say or think when she denies that she is turning '
            'to religion?'
        )
    ),
    "nurse-love-addiction-R0106": (
        (
            "What does Itsuki say while describing the club's cosplay activities "
            "and Asuka's potential appeal to nerds?"
        )
    ),
    "nurse-love-addiction-R0107": (
        (
            "What does Itsuki say while explaining why Sakuya's boarding-school "
            'choice shows more than simple concern for her mother?'
        )
    ),
    "nurse-love-addiction-R0109": (
        (
            "What else does Sakuya say while joining the students' lively beach "
            'conversation?'
        )
    ),
    "nurse-love-addiction-R0110": (
        'What else does Itsuki say while inviting Asuka and Nao to the seaside?'
    ),
    "nurse-love-addiction-R0111": (
        (
            'What other study instructions does Kaede give while explaining the '
            'purpose of the vacation review tests?'
        )
    ),
    "nurse-love-addiction-R0112": (
        (
            'What else does Asuka say or think during the conversation in which she '
            'insists that she really wants to change?'
        )
    ),
    "nurse-love-addiction-R0113": (
        (
            'What else does Asuka say or think while discussing how Itsuki and '
            'Sakuya got together?'
        )
    ),
    "heart-of-the-woods-R0071": (
        "After Tara forces open the church doors, what does she do when the monster "
        "appears?"
    ),
    "heart-of-the-woods-R0072": (
        "After Tara eagerly awaits Morgan's secret, what does she do as they set "
        "off?"
    ),
    "heart-of-the-woods-R0073": (
        "What does Tara do after hurrying beside Morgan toward their destination?"
    ),
    "heart-of-the-woods-R0074": (
        "After Morgan smiles at Tara, what comforting gesture does she later make?"
    ),
    "heart-of-the-woods-R0075": (
        "After Morgan takes the carriage reins, what does she do when she is asked a "
        "question?"
    ),
    "heart-of-the-woods-R0076": (
        "After Abigail approaches the forest spirit, what does she do when it becomes "
        "a tree?"
    ),
    "heart-of-the-woods-R0078": (
        "After Abigail lets the fawn approach her, what does she do afterward?"
    ),
    "heart-of-the-woods-R0080": (
        "After Tara grins at a new audience, what does she do as she prepares to "
        "talk?"
    ),
    "heart-of-the-woods-R0084": (
        "After Tara tries to film the monster, what does she do later in the story?"
    ),
    "heart-of-the-woods-R0086": (
        "Which two moments involving Tara occur in the same scene?"
    ),
    "heart-of-the-woods-R0089": (
        "In what order do these moments in Tara's story occur?"
    ),
    "heart-of-the-woods-R0090": (
        "Arrange these milestones in Tara's story from earliest to latest?"
    ),
    "heart-of-the-woods-R0093": (
        "After Tara lifts her bags for Morgan, what does she do next?"
    ),
    "heart-of-the-woods-R0094": (
        "In what order do these events in Tara's story occur?"
    ),
    "heart-of-the-woods-R0099": (
        "Put these developments in Tara's story in chronological order?"
    ),
    "heart-of-the-woods-R0100": (
        "Arrange these milestones in Abigail's story from earliest to latest?"
    ),
    "heart-of-the-woods-R0101": (
        "From earliest to latest, order these developments in Tara's story?"
    ),
    "nurse-love-addiction-R0028": (
        "In what order do these milestones in Sakuya and Itsuki's relationship occur?"
    ),
    "nurse-love-addiction-R0036": (
        "Arrange these moments involving Asuka from earliest to latest?"
    ),
    "nurse-love-addiction-R0037": (
        "From earliest to latest, how do these moments involving Itsuki and Nao "
        "unfold?"
    ),
    'nurse-love-addiction-R0038': "In what order do these moments from Asuka's reflections and conversations with Nao and Itsuki occur?",
    'nurse-love-addiction-R0039': "In what order do these moments from Asuka's training and interactions with Nao and Itsuki occur?",
    'nurse-love-addiction-R0041': "In what order do these moments involving Asuka's dorm life and her friends occur?",
    'nurse-love-addiction-R0043': "In what order do Asuka's remarks and thoughts during her time with Nao occur?",
    'nurse-love-addiction-R0045': 'In what order do these reflections and exchanges involving Asuka, Itsuki, and Nao occur?',
    'nurse-love-addiction-R0047': 'In what order do these moments involving Kaede, Nao, and Asuka occur?',
    "nurse-love-addiction-R0040": (
        "Arrange these moments involving Asuka and Itsuki from earliest to latest?"
    ),
    "nurse-love-addiction-R0042": (
        "From earliest to latest, how do these moments involving Asuka and Kaede "
        "unfold?"
    ),
    "nurse-love-addiction-R0044": (
        "From earliest to latest, how do these moments involving Asuka and Nao "
        "unfold?"
    ),
    "nurse-love-addiction-R0046": (
        "Arrange these moments involving Itsuki and Kaede from earliest to latest?"
    ),
    "nurse-love-addiction-R0048": (
        "From earliest to latest, how do these moments involving Asuka and Itsuki "
        "unfold?"
    ),
    "fault-milestone-two-R0006": (
        "What best describes how Melano's confrontation with the group develops?"
    ),
    "fault-milestone-two-R0008": (
        "What does Melano tell the group about stopping the attacks, their apparent "
        "deaths, and the danger of returning home?"
    ),
    "fault-milestone-two-R0029": (
        "What do Selphine and Ritona realize about their surroundings and the group's "
        "readiness for a fight during the voyage?"
    ),
    "fault-milestone-two-R0037": (
        "What do Selphine and Ritona conclude about converting food into mana and "
        "committing that mana in Neo Sasary?"
    ),
    "fault-milestone-two-R0070": (
        "What does Sol explain about Serisian burial soil and where the group can "
        "obtain fertile soil?"
    ),
    "fault-milestone-two-R0081": (
        "What do Rupika and Selphine say about the Vita Domain and Ritona's medicine "
        "after Ritona collapses?"
    ),
    "fault-milestone-two-R0096": (
        "What does Ritona say about Greus and the deadline for his promise?"
    ),
    "fault-milestone-two-R0104": (
        "What do the Guild Newbie, the pharmacist, and Selphine reveal about treating "
        "the invisible noose and selling sagiolla?"
    ),
    "highway-blossoms-R0011": (
        "How does Amber and Marina's first encounter develop into the beginning of "
        "their treasure hunt?"
    ),
    "highway-blossoms-R0029": (
        "How does the group's visit to Canyon de Chelly unfold?"
    ),
    "highway-blossoms-R0041": (
        "How does Marina respond to Amber's fatigue and to the food cooling during "
        "their drive?"
    ),
    "highway-blossoms-R0043": (
        "How does Joseph explain why his partnership with Mariah works despite their "
        "differences?"
    ),
    "highway-blossoms-R0045": None,
    "highway-blossoms-R0055": (
        "How do the travelers' relationships develop during their shared sightseeing?"
    ),
    "highway-blossoms-R0072": (
        "How do Amber and Marina's conversations during their visit to Zion unfold?"
    ),
    "highway-blossoms-R0084": (
        "How do Marina and Amber's limited dating histories differ, and why had Amber "
        "avoided serious relationships?"
    ),
    "highway-blossoms-R0085": (
        "How does Amber respond after learning that Marina lost her share of the "
        "treasure?"
    ),
    "highway-blossoms-R0094": (
        "How do Amber and Marina's conversations in the lead-up to the festival unfold?"
    ),
    "fault-milestone-two-R0005": (
        "What do Selphine and Melano reveal when Selphine introduces herself and "
        "Melano assesses her symptoms?"
    ),
    "fault-milestone-two-R0013": (
        "How does the opening confrontation among Selphine, Melano, and Reighnvhasta unfold?"
    ),
    "fault-milestone-two-R0016": (
        "How does Melano's confrontation with Selphine's group reach its conclusion?"
    ),
    "fault-milestone-two-R0018": (
        "How does the group regroup once Selphine returns to herself?"
    ),
    "fault-milestone-two-R0021": (
        "How does Ritona's explanation of the Path-down to Rune unfold?"
    ),
    "fault-milestone-two-R0022": (
        "How does Riggs's conversation with Flora about Rughzenhaide unfold?"
    ),
    "fault-milestone-two-R0030": (
        "How does the group's first voyage through unfamiliar surroundings unfold?"
    ),
    "fault-milestone-two-R0035": (
        "How does Selphine and Ritona's discussion of food and mana develop?"
    ),
    "fault-milestone-two-R0039": (
        "What does the owner ask the group to do about Sol?"
    ),
    "fault-milestone-two-R0040": (
        "What do the innkeeper and Sol say about whether the group can stay?"
    ),
    "fault-milestone-two-R0044": (
        "How does the conversation about Sol joining the group unfold over dinner?"
    ),
    "fault-milestone-two-R0045": (
        "Whom does Ritona consider consulting when the group discusses finding medical care?"
    ),
    "fault-milestone-two-R0046": (
        "Which statements accurately summarize Volthal and Flora's report about the "
        "missing pair?"
    ),
    "fault-milestone-two-R0047": (
        "How does the discussion of the Inner-Pole and the missing pair unfold?"
    ),
    "fault-milestone-two-R0053": (
        "How does the group's search for medical help and supplies unfold after they "
        "meet Sol?"
    ),
    "fault-milestone-two-R0055": (
        "How does the lodging dispute involving Sol and the girls unfold?"
    ),
    "fault-milestone-two-R0058": (
        "How does Selphine conduct her investigation involving Sol?"
    ),
    "fault-milestone-two-R0061": (
        "How does the search for a missing child at the bathhouse escalate?"
    ),
    "fault-milestone-two-R0062": (
        "How does the confrontation over Selphine's mind dive and Sol unfold?"
    ),
    "fault-milestone-two-R0065": (
        "What does Sol ask the group to help arrange for Mil?"
    ),
    "fault-milestone-two-R0067": (
        "What do Sol and Selphine say about Mil's crisis and Greus's past?"
    ),
    "fault-milestone-two-R0074": (
        "Where does Sol say the group can find fertile soil?"
    ),
    "fault-milestone-two-R0075": (
        "How does the group discuss Serisian soil and the search for fertile soil?"
    ),
    "fault-milestone-two-R0076": (
        "How does the conversation about seeking Greus's help for Mil unfold?"
    ),
    "fault-milestone-two-R0080": (
        "How does Sceatoire's confrontation with Selphine and Rune unfold?"
    ),
    "fault-milestone-two-R0088": (
        "How does Selphine assess whether Greus has kept his word?"
    ),
    "fault-milestone-two-R0090": (
        "How does the discussion of Ritona's condition and Greus's demands unfold?"
    ),
    "fault-milestone-two-R0091": (
        "How does Rupika explain the role of the Vita Domain in Ritona's care?"
    ),
    "fault-milestone-two-R0098": (
        "How does the group discuss spreading news and the consequences it may bring?"
    ),
    "fault-milestone-two-R0102": (
        "How do Ritona and Selphine's views of Greus develop during their discussion?"
    ),
    "fault-milestone-two-R0106": (
        "How does the conversation about Mil's article and special maytaux dish unfold?"
    ),
    "fault-milestone-two-R0107": (
        "How do Sol and Mil respond to the pharmacist's actions?"
    ),
    "fault-milestone-two-R0108": (
        "What does Mil tell Sol about killing the man responsible for her condition?"
    ),
    "fault-milestone-two-R0111": (
        "How does the discussion expose the truth about the invisible noose and sagiolla?"
    ),
    "fault-milestone-two-R0112": (
        "How does the confrontation among Sol, Mil, Selphine, and the pharmacist unfold?"
    ),
    "highway-blossoms-R0010": (
        "How does Amber and Marina's relationship develop across their trip?"
    ),
    "highway-blossoms-R0014": (
        "How does their first attempt to interpret the treasure clues unfold?"
    ),
    "highway-blossoms-R0019": (
        "How does Amber and Marina's encounter with Mariah's group near the store unfold?"
    ),
    "highway-blossoms-R0021": (
        "How does Amber and Marina's search planning near Roswell unfold?"
    ),
    "highway-blossoms-R0022": (
        "How does the story establish Mariah's relationships with Joseph and Tess?"
    ),
    "highway-blossoms-R0023": (
        "How does the conversation about Marina's car and the treasure search unfold?"
    ),
    "highway-blossoms-R0027": (
        "How does the exchange about the missing car and Mariah's group unfold?"
    ),
    "highway-blossoms-R0030": (
        "Which sequence correctly orders the travelers' remarks during discussions "
        "of destinations and personal goals?"
    ),
    "highway-blossoms-R0031": (
        "How do Marina's questions about the Grand Canyon recur during the treasure search?"
    ),
    "highway-blossoms-R0036": (
        "How do Amber and Marina's encounters with strangers during the journey unfold?"
    ),
    "highway-blossoms-R0038": (
        "How does the group's visit and search at the ruins unfold?"
    ),
    "highway-blossoms-R0039": (
        "How does Amber and Marina's conversation during the drive to the Grand Canyon develop?"
    ),
    "highway-blossoms-R0047": (
        "How do the group's interactions develop during their shared sightseeing?"
    ),
    "highway-blossoms-R0050": (
        "How does the group's attitude toward taking photographs change during the trip?"
    ),
    "highway-blossoms-R0051": (
        "How does the story establish the sibling dynamic among Mariah, Joseph, and Tess?"
    ),
    "highway-blossoms-R0057": (
        "How does Amber and Marina's diner conversation unfold?"
    ),
    "highway-blossoms-R0061": (
        "What is the chronology of Amber and Marina's stop in town and exchange with Mariah?"
    ),
    "highway-blossoms-R0062": (
        "How does Amber and Marina's uncomfortable rest-stop visit unfold?"
    ),
    "highway-blossoms-R0067": (
        "Which statements accurately describe Amber and Marina's decisions as they "
        "prepare to leave Arches?"
    ),
    "highway-blossoms-R0069": (
        "How does the group's discussion at Angel's Landing unfold as they reassess the clue?"
    ),
    "highway-blossoms-R0071": (
        "Which sequence best traces Amber's changing sense of family?"
    ),
    "highway-blossoms-R0073": (
        "How do Amber and Marina respond to setbacks around Angel's Landing?"
    ),
    "highway-blossoms-R0075": (
        "How does Amber and Marina's conversation during their approach to Las Vegas develop?"
    ),
    "highway-blossoms-R0077": (
        "How does Amber and Marina's first day in Las Vegas unfold?"
    ),
    "highway-blossoms-R0081": (
        "How does Amber and Marina's afternoon in Las Vegas unfold?"
    ),
    "highway-blossoms-R0083": (
        "How does Amber's relationship with her memories of Gramps develop over the journey?"
    ),
    "highway-blossoms-R0088": (
        "How does the group respond to the motorhome breakdown?"
    ),
    "highway-blossoms-R0091": (
        "How does the conversation leading into Amber's wager with Mariah unfold?"
    ),
    "highway-blossoms-R0096": (
        "How does Amber and Marina's argument after the engine trouble unfold?"
    ),
    "highway-blossoms-R0097": (
        "How does Amber's plan to recover Marina's share play out at blackjack?"
    ),
    "highway-blossoms-R0099": (
        "How do Amber and Marina's conversations about music develop across the trip?"
    ),
    "highway-blossoms-R0101": (
        "How does Mariah and Amber's diner confrontation unfold?"
    ),
    "highway-blossoms-R0104": (
        "How does Amber and Marina's reconciliation at the hotel unfold?"
    ),
    "highway-blossoms-R0106": (
        "How does the final stretch of the treasure hunt and the farewell with Joseph's "
        "group unfold?"
    ),
    "fata-morgana-requiem-R0003": (
        "Which sequence best traces Michel and Imeon's debate over survival?"
    ),
    "fata-morgana-requiem-R0006": (
        "Reconstruct the progression of Imeon's conversation about Danish seafaring."
    ),
    "fata-morgana-requiem-R0012": (
        "Place the listed turning points from Imeon's first extended conversation "
        "with Michel in story order."
    ),
    "fata-morgana-requiem-R0013": (
        "Which sequence best traces Imeon's first visit to Michel's mansion?"
    ),
    "fata-morgana-requiem-R0015": (
        "From the options, identify the progression of Imeon's debate with Michel "
        "about adventure."
    ),
    "fata-morgana-requiem-R0019": (
        "Reconstruct the progression of Imeon's opening storyline at the mansion."
    ),
    "fata-morgana-requiem-R0022": (
        "Place the listed turning points from Imeon's growing entanglement at the "
        "mansion in story order."
    ),
    "fata-morgana-requiem-R0026": (
        "From the options, identify the progression of the commotion over an "
        "unexpected mansion visitor."
    ),
    "fata-morgana-requiem-R0030": (
        "Which sequence best traces Imeon's evolving outlook during his conversation "
        "with Michel?"
    ),
    "fata-morgana-requiem-R0033": (
        "Reconstruct the progression of Michel's attempt to handle the unexpected visitor."
    ),
    "fata-morgana-requiem-R0034": (
        "Place the listed turning points from the visitor's exchange with Michel in "
        "story order."
    ),
    "fata-morgana-requiem-R0037": (
        "From the options, identify the progression of the visitor's disruption at "
        "the mansion."
    ),
    "fata-morgana-requiem-R0038": (
        "Arrange the key exchanges in Michel's response to the intrusion in story order."
    ),
    "fata-morgana-requiem-R0039": (
        "Track the changes in the mansion's atmosphere during the visitor's arrival."
    ),
    "fata-morgana-requiem-R0041": (
        "Which ordering best captures the household's response to the unexpected guest?"
    ),
    "fata-morgana-requiem-R0043": (
        "Put the developments from Georges's studio commotion in story order."
    ),
    "fata-morgana-requiem-R0045": (
        "Arrange the stages of Imeon's early mansion arc in story order."
    ),
    "fata-morgana-requiem-R0048": (
        "Track the changes in Mell's bond with Morgana across their conversation."
    ),
    "fata-morgana-requiem-R0051": (
        "Which sequence best traces Morgana's first night at the estate?"
    ),
    "fata-morgana-requiem-R0055": (
        "Reconstruct the progression of Morgana's doorway encounter."
    ),
    "fata-morgana-requiem-R0060": (
        "Place the listed turning points from a period of strain at the estate in "
        "story order."
    ),
    "fata-morgana-requiem-R0064": (
        "From the options, identify the progression of Morgana's search for security "
        "at the estate."
    ),
    "fata-morgana-requiem-R0068": (
        "Arrange the developments in Morgana's departure from the great hall in story order."
    ),
    "fata-morgana-requiem-R0070": (
        "Which sequence best traces Jacopo's protective role toward Morgana?"
    ),
    "fata-morgana-requiem-R0073": (
        "Reconstruct the progression of Morgana's early bonds at the estate."
    ),
    "fata-morgana-requiem-R0075": (
        "Place the listed turning points from a tense night at the estate in story order."
    ),
    "fata-morgana-requiem-R0076": (
        "From the options, identify the progression of Morgana's vulnerability during "
        "her early stay."
    ),
    "fata-morgana-requiem-R0080": (
        "Arrange the developments in the estate's escalating unrest in story order."
    ),
    "fata-morgana-requiem-R0084": (
        "Track the changes in the estate's atmosphere of crisis across this passage."
    ),
    "fata-morgana-requiem-R0088": (
        "Track the changes in tone as Michel's movie date with Giselle begins."
    ),
    "fata-morgana-requiem-R0090": (
        "Which ordering best captures the couple's reaction to the horror film?"
    ),
    "fata-morgana-requiem-R0094": (
        "Which ordering best captures the overall shape of Michel and Giselle's movie date?"
    ),
    "fata-morgana-requiem-R0096": (
        "Put the developments concerning the couple's attempt to process the horror film in "
        "story order."
    ),
    "fata-morgana-requiem-R0097": (
        "Which sequence best traces the couple's commitment during their reunion conversation?"
    ),
    "fata-morgana-requiem-R0098": (
        "Put the developments from the closing phase of the date in story order."
    ),
    "fata-morgana-requiem-R0101": (
        "Reconstruct the progression of Michel's choice about a future with Giselle."
    ),
    "fata-morgana-requiem-R0104": (
        "Place the listed turning points from the couple's discussion of identity in "
        "story order."
    ),
    "fata-morgana-requiem-R0108": (
        "From the options, identify the progression of Michel's invitation to Giselle."
    ),
    "fata-morgana-requiem-R0111": (
        "Arrange the decisions shaping Michel's readiness to build a life with Giselle "
        "in story order."
    ),
    "fata-morgana-requiem-R0113": (
        "Track the changes in Michel's thinking during the post-film conversation."
    ),
    "fata-morgana-requiem-R0119": (
        "Which ordering best captures the emotional arc of the movie outing?"
    ),
    "fata-morgana-requiem-R0121": (
        "Put the developments concerning Michel's response to the film in story order."
    ),
    "fata-morgana-requiem-R0124": (
        "Arrange the exchanges in the couple's reflection on living again in story order."
    ),
    "fata-morgana-requiem-R0127": (
        "Track the changes in the date's reflective tone across the conversation."
    ),
    "fata-morgana-requiem-R0131": (
        "Follow the arc of the couple's shared-future conversation by choosing the "
        "correct sequence."
    ),
    "fata-morgana-requiem-R0137": (
        "Which ordering best captures Morgana's approach to Midsummer?"
    ),
    "fata-morgana-requiem-R0140": (
        "Which ordering best captures Morgana's final confrontation over the illusion?"
    ),
    "fata-morgana-requiem-R0143": (
        "Put the developments concerning Morgana's effort to recover missed experiences in "
        "story order."
    ),
    "fata-morgana-requiem-R0151": (
        "Put the developments concerning Morgana's adjustment to the idealized realm "
        "in story order."
    ),
    "fata-morgana-requiem-R0155": (
        "Identify the story order of the developments shaping Morgana's view of her "
        "lost childhood."
    ),
    "fata-morgana-requiem-R0158": (
        "Identify the story order of the developments in Morgana's pursuit of an "
        "ordinary life."
    ),
    "fata-morgana-requiem-R0162": (
        "Identify the story order of the developments shaping Morgana's engagement "
        "with the peaceful realm."
    ),
    "fata-morgana-requiem-R0164": (
        "Identify the story order of the developments in Morgana's outlook on Midsummer."
    ),
    "fata-morgana-requiem-R0169": (
        "Identify the story order of the developments shaping Morgana's view of the "
        "idealized realm."
    ),
    "fata-morgana-requiem-R0173": (
        "Follow the arc of Morgana's outlook during the peaceful interlude by choosing "
        "the correct sequence."
    ),
    "fata-morgana-requiem-R0176": (
        "Follow the arc of Morgana's reassessment of her companion by choosing the "
        "correct sequence."
    ),
    "fata-morgana-requiem-R0180": (
        "Follow the arc of Morgana's relationship in the peaceful realm by choosing "
        "the correct sequence."
    ),
}


OPTION_REWRITES: dict[str, dict[str, str]] = {'9-nine-episode-1-R0046': {'A': 'They later share routine dinners for two.'},
 '9-nine-episode-1-R0047': {'A': 'She says she does not believe the sacred-relic legend.',
                            'C': 'She recommends not worrying about paranormal phenomena.',
                            'D': 'She later teaches the White Serpent relic lore in class.'},
 '9-nine-episode-1-R0048': {'B': 'She describes allowing herself a monthly splurge.',
                            'C': 'She later warns Kakeru against reckless crane-game spending.',
                            'D': 'She calls an almost-800-yen combo expensive on her first date.'},
 '9-nine-episode-1-R0049': {'B': 'She asks to call him Kakeru-kun.',
                            'D': 'She accidentally says Niimi-kun again and corrects herself.'},
 '9-nine-episode-1-R0050': {'C': 'He says his bought lunch is onigiri.'},
 '9-nine-episode-1-R0051': {'C': 'He introduces himself to Haruka at Nine Ball.'},
 '9-nine-episode-1-R0052': {'A': 'She is later identified as the User Noa Yuuki.',
                            'B': 'After the operation she reminds him that he owes the week.',
                            'C': 'She accepts the one-week-parfait deal.'},
 '9-nine-episode-1-R0053': {'A': 'Later he renews the promise to protect her no matter what.'},
 '9-nine-episode-1-R0057': {'D': "His total reaches 2,000 yen despite Miyako's objections.",
                            'E': 'He inserts one more 100 yen.'},
 'heart-of-the-woods-R0002': {'A': 'When Madison first calls Abigail her girlfriend.',
                              'B': 'Before their first kiss.',
                              'C': 'Immediately after their first kiss.',
                              'D': 'During their first kiss.',
                              'E': 'Only after the final confrontation.'},
 'heart-of-the-woods-R0003': {'A': 'One hundred meters.',
                              'B': 'One mile.',
                              'C': 'Half a mile.',
                              'D': 'Two miles.',
                              'E': 'Three miles.'},
 'heart-of-the-woods-R0004': {'A': 'Five minutes.',
                              'B': 'Thirty minutes.',
                              'C': 'Ten minutes.',
                              'D': 'One minute.',
                              'E': 'Two minutes.'},
 'heart-of-the-woods-R0033': {'A': 'Girlfriends.',
                              'B': 'Platonic friends.',
                              'C': 'Casual partners.',
                              'E': 'Friends with benefits.',
                              'F': 'Dating partners.'},
 'heart-of-the-woods-R0036': {'A': 'Dating.',
                              'B': 'Lovers.',
                              'D': 'Girlfriends.',
                              'E': 'Close friends who had agreed to remain platonic.',
                              'F': 'Partners.'},
 'heart-of-the-woods-R0041': {'A': 'Weakened fairy blood.',
                              'B': 'A lunar barrier around Evelyn.',
                              'C': "A rule tied to the old queen's crown.",
                              'D': 'A broken seasonal pact.',
                              'F': 'Reduced forest mana.'},
 'heart-of-the-woods-R0044': {'A': 'The winter queen.',
                              'B': 'A former forest guardian.',
                              'C': 'The spring queen.',
                              'F': 'A named rogue called Frio.'},
 'heart-of-the-woods-R0046': {'A': 'Six.', 'B': 'Two.', 'C': 'Three.', 'E': 'Five.', 'F': 'Four.'},
 'heart-of-the-woods-R0048': {'A': 'Because the previous host was dying.',
                              'B': 'To gain political authority.',
                              'C': 'To control access to the forest.',
                              'D': 'Because the body had fairy blood.',
                              'F': 'To get close to Morgan.'},
 'heart-of-the-woods-R0050': {'A': 'A seasonal oath.',
                              'B': "The queen's true name.",
                              'D': 'A Latin prayer.',
                              'F': 'A three-word fairy phrase.'},
 'heart-of-the-woods-R0053': {'A': 'Historian.',
                              'C': 'Journalist.',
                              'D': 'Video editor.',
                              'E': 'Paranormal investigator for another show.',
                              'F': 'Teacher.'},
 'heart-of-the-woods-R0055': {'B': 'Five centuries.',
                              'C': 'Fifteen centuries.',
                              'D': 'Ten centuries.',
                              'E': 'Three centuries.',
                              'F': 'Two centuries.'},
 'heart-of-the-woods-R0056': {'F': 'The forest spirit during the storm.'},
 'heart-of-the-woods-R0057': {'B': 'Adrenaline.',
                              'C': 'Fairy magic.',
                              'D': 'The curse.',
                              'E': 'Caffeine.',
                              'F': 'The forest.'},
 'nurse-love-addiction-R0001': {'A': 'Asuka says it sounds fun. Asuka suggests that they do it. Asuka '
                                     'asks whether the plan would satisfy Sakuya.',
                                'E': "Asuka says it's true that it's completely white, but it's "
                                     'not a nurse outfit.'},
 'nurse-love-addiction-R0002': {'A': 'This time, Miss Sakuya will also be accompanying Ms. '
                                     'Shionogi.',
                                'B': "Miss Sakuya's bright-red face is a rare sight.",
                                'C': 'Her voice is surprisingly gentle and she cannot say anything '
                                     'in response.',
                                'D': 'Asuka excitedly says that she loves her very much.',
                                'E': 'Miss Itsuki waves her hand at her again and walks away.'},
 'nurse-love-addiction-R0003': {'A': 'Itsuki says of course, her lady. Itsuki asks whether they '
                                     'would like to request it.',
                                'B': "Itsuki says its content's been corrupted. Itsuki wonders "
                                     'what is happening.',
                                'C': "Itsuki says if that's so. Itsuki says hm, no, she guesses "
                                     "that's not possible.",
                                'D': "Itsuki says they do, they do. Itsuki says they just don't "
                                     'realize it.',
                                'E': "Itsuki says at least not today. Itsuki says it's all in good "
                                     'fun.'},
 'nurse-love-addiction-R0004': {'C': 'Picking up the smartphone, she examines the screen. Silence. '
                                     "This isn't good.",
                                'D': 'During the time she remains in a daze, Ms. Ohara disappears '
                                     'from sight.',
                                'E': 'Asuka cannot hear anything. Asuka cannot feel anything but '
                                     'her consciousness ebbing away.'},
 'nurse-love-addiction-R0005': {'B': 'The two easygoing friends chime in, leaving Asuka frustrated by how easy it is for them to say that.',
                                'D': 'As she asks her this, Miss Itsuki keeps a long silence '
                                     'before responding.',
                                'E': 'Still worried about Nao, she cannot seem to think of any '
                                     'polite reply.'},
 'nurse-love-addiction-R0006': {'A': 'Nao says the situation escalated quickly, then offers her support and encourages her sister to do her best.',
                                'B': 'Nao says that she rushed home because she was worried about her sister and asks what she is doing.',
                                'C': 'Nao says another message is from Itsuki and reads, ‘Say goodnight to Prima for me.’',
                                'E': 'Nao warns her sister that if she does not wake up, Nao is going to kiss her.'},
 'nurse-love-addiction-R0007': {'A': 'Itsuki guarantees that Asuka has the potential to become a star among nerds.',
                                'C': "Itsuki says having such a ‘power' would be like having the "
                                     'power of a god.',
                                'D': "Itsuki says it won't just be that Itsuki will do this and "
                                     'that to them.',
                                'E': "Itsuki says it's all a performance. Itsuki says people like "
                                     "these, they have got to show them who's boss."},
 'nurse-love-addiction-R0008': {'A': 'Kaede looks forward to getting to know them all. Later, '
                                     'Kaede says bedridden patients and patients with delicate '
                                     'skin will quickly form bedsores. Kaede adds that this will '
                                     'lead to more serious problems.',
                                'B': "Kaede says it's true that her elder sister, Sumire Ohara, "
                                     'won the Nightingale Award. Later, Kaede looks forward to '
                                     'getting to know them all.',
                                'D': "Kaede explains that bedridden patients and those with delicate skin quickly develop bedsores, which can lead to more serious problems. Later, she says that small things can intensify an already nervous patient's anxiety.",
                                'E': "Kaede says that small things can intensify an already nervous patient's anxiety. Later, she confirms that her elder sister, Sumire Ohara, won the Nightingale Award."},
 'nurse-love-addiction-R0009': {'A': 'Asuka says that she is basically the same age as their grandmother.',
                                'B': 'Asuka interrupts and asks whether this is what they are looking for.',
                                'D': 'Asuka apologizes for spacing out, then laughs softly.'},
 'nurse-love-addiction-R0010': {'A': 'Nao says now, milk and eggs go into the fridge.',
                                'B': 'Nao tells Asuka that they took a while and asks whether they '
                                     'went far.',
                                'D': 'Nao says that someone was there even though the room was empty.',
                                'E': 'Nao wonders whether it could be a stalker.'},
 'nurse-love-addiction-R0011': {'B': 'The classroom is engulfed with cries of despair.',
                                'C': 'The more she thinks of it the less she understands.',
                                'D': 'Asuka turns away and ignores her. Eeek, mission failed.'},
 'nurse-love-addiction-R0012': {'B': 'She fears that it may already be closed; just then, something happens.',
                                'C': 'Miss Sakuya freezes as though time itself has stopped.'},
 'nurse-love-addiction-R0013': {'B': 'Nao says sis, she is acting like a heckler with all her '
                                     'little jibes.',
                                'C': "Nao says not to worry because she practiced a lot after school.",
                                'D': 'Nao reminds her sister that they learned this in nutrition class two weeks earlier.',
                                'E': "Nao says it's obvious, though, why she wasn't in any of the "
                                     "laboratory's data."},
 'nurse-love-addiction-R0014': {'A': "Asuka explains that she came chasing after them in a hurry.",
                                'C': 'Asuka says that Miss Takeda and Miss Amato have healing hands.',
                                'D': "Asuka envies the nurse's cap and the candlelight service.",
                                'E': 'Asuka playfully tells Nao that she woke up on her own, then pretends to sleep.'},
 'nurse-love-addiction-R0015': {'B': "Ms. Ohara's words flow, her body never taking a moment to "
                                     'rest as she moves serenely and gracefully.',
                                'C': 'Swoosh. Crash. The sounds of the waves reach her ears. Asuka '
                                     'remains dazed, her eyes round with surprise.',
                                'D': 'Perhaps because she had looked so pensive when she told her, '
                                     "a downcast expression appears on Nao's face.",
                                'E': 'Asuka hears Miss Takeda whisper quietly. Responding to her '
                                     'voice, she looks towards her. Their eyes meet.'},
 'nurse-love-addiction-R0016': {'D': 'Asuka says that she and the others will go through a lot '
                                     'together over the next three years.',
                                'E': 'Asuka says it is not a big deal, even if they are lost forever.'},
 'nurse-love-addiction-R0017': {'B': 'Once again, excited voices can be heard. There truly are a '
                                     'lot of events to come.',
                                'D': 'Miss Sakuya responds coldly. Everyone seems to agree with '
                                     'this assessment. Machi and Michi are nodding.',
                                'E': "Ms. Ohara's kind smile makes her weak in the knees, unlike her reaction to Nao."},
 'nurse-love-addiction-R0019': {'A': 'Asuka broaches the subject as if to suggest that they could do it if Nao wanted, though it feels strange because Nao had recommended medicine instead.',
                                'B': "Itsuki says it's what Prima looks like after she's "
                                     'transformed. Itsuki says when she first saw them at school, '
                                     'she felt shockwaves.',
                                'C': 'Nao goes back into the kitchen as she speaks.',
                                'D': 'Asuka opens her photo folder, chooses the oldest picture, and taps it with her finger.'},
 'nurse-love-addiction-R0021': {'A': 'Next to her, Nao is quietly scribbling in her notebook.',
                                'B': 'Ms. Ohara looks at Asuka as though she does not quite understand what Asuka means.',
                                'C': 'Itsuki says perhaps having suffered traumatic experiences in '
                                     "the orphanage, she's lost her memories from then.",
                                'D': 'Sakuya acknowledges that Asuka can hear them and does not appear out of place.',
                                'F': 'Hearing these words from Nao, she begins to think that might '
                                     'be the case.'},
 'nurse-love-addiction-R0023': {'A': 'Environment Theory, Pathology, Microbiology, Clinical '
                                     'Testing, Information Science, Counseling. Her mind spirals.',
                                'B': 'Itsuki says that all she wants is to make Prima into something great, then agrees to see them.',
                                'C': 'Asuka says words like spec and data are from that, and the '
                                     'names she did not know are her gaming friends.',
                                'D': 'Nao looks deeply relieved, from the bottom of her heart.',
                                'E': 'Itsuki tells Hiromi to stop messing with her and to make sure '
                                     'the script is followed.',
                                'F': "Sakuya says the only reason why she hasn't returned to the "
                                     "dorm is that she'd rather not be around her and her "
                                     'nagging.'},
 'nurse-love-addiction-R0024': {'A': 'Asuka says that she does not think they can see it, even if they look very carefully.',
                                'C': 'Asuka decides that the answer is the thigh and privately celebrates her victory over Nao.',
                                'D': 'Nao begins to describe all the tests that were performed on her and the others.',
                                'E': 'Nao calls her a cruel girl and says that she is not really their sister.',
                                'F': 'Led by Ms. Shionogi, they leave the nurse station. The '
                                     "hospital ward they head to isn't the one where Yuki is."},
 'nurse-love-addiction-R0036': {'C': 'Ms. Ohara nods, but her expression seems to say otherwise. '
                                     'Asuka does not seem convinced.',
                                'D': 'Meaningless thoughts about the former Jellyfish Club manager dance through her head.'},
 'nurse-love-addiction-R0037': {'A': 'Nao says if they plan properly and work at it consistently '
                                     'then they will be able to get through it. Nao says leave it '
                                     'to her.',
                                'B': 'Itsuki handles them independently and says that they are slightly naughty by nature.',
                                'C': "Sakuya says it's not like they are together all the time. "
                                     'Sakuya says today, a bunch of girls from class took her off '
                                     'somewhere.',
                                'D': 'Itsuki says that it took a little coercion.'},
 'nurse-love-addiction-R0038': {'B': 'As she asks her this, Miss Itsuki keeps a long silence '
                                     'before responding.',
                                'C': "Asuka finds her past and Miss Sakuya's role confusing after everything that has happened.",
                                'D': "Asuka involuntarily lets out an odd cry, quickly covers her mouth, and hears only Nao's quiet breathing."},
 'nurse-love-addiction-R0039': {'A': 'Nao breaks the tension. Miss Itsuki shrugs her shoulders '
                                     'apologetically.',
                                'C': 'Asuka acknowledges the instruction and asks the patient to hold out an arm.'},
 'nurse-love-addiction-R0040': {'A': 'Itsuki says an event that is said to be the biggest doujin '
                                     'event in the country.',
                                'C': 'Without any further words, she enters the classroom.',
                                'D': 'Kaede says this decision has been made upon receiving a '
                                     'formal request from the chief nurse. Kaede asks whether they '
                                     'know what this means.'},
 'nurse-love-addiction-R0041': {'A': 'Itsuki says that her words told them to do the same.',
                                'B': 'After school, she returns to the dorm alone.',
                                'D': 'Nao says when she was cleaning the room she found it on the '
                                     'floor.'},
 'nurse-love-addiction-R0042': {'A': 'Asuka playfully tells Nao that she woke up on her own, then pretends to sleep.',
                                'B': "A click and approaching footsteps bring Asuka's senses slowly back to her.",
                                'C': 'Asuka says that they were not in the first-year classroom.',
                                'D': 'Kaede says that the ceremony will be held in the auditorium and that everyone will attend in training uniforms.'},
 'nurse-love-addiction-R0043': {'B': 'Asuka says beyond the hill, she sees them, she sees them '
                                     'gooooo.',
                                'D': 'Asuka says it seems like she is taking that medicine quite '
                                     'often.'},
 'nurse-love-addiction-R0044': {'A': 'Nao asks not to be patronized, reminding her sister that they have been together for a long time.',
                                'B': 'Asuka slowly turns her eyes toward the paper and cries, ‘Burn!’',
                                'D': 'Sakuya tells Itsuki not to say unnecessary things.'},
 'nurse-love-addiction-R0045': {'A': "Itsuki says it would've made her look a lot cooler if her "
                                     'reason for collapsing had been too much studying.',
                                'B': 'Asuka imagines a god who could heal any wound and even bring the dead back to life.',
                                'C': 'Itsuki says that some children unknown to her might nevertheless have known her.',
                                'D': 'Nao calls herself an evil, cruel girl who did unforgivable things and tells Asuka to do as she pleases.'},
 'nurse-love-addiction-R0046': {'A': 'Kaede says that she is not worth choosing as a role model.',
                                'B': 'Sakuya wears a dress that is not particularly suitable for hospital life.',
                                'C': 'Itsuki says of course, her lady. Itsuki asks whether they '
                                     'would like to request it.',
                                'D': 'For a moment, she speaks in a questioning voice, but it '
                                     'quickly turns to a gasp of surprise.'},
 'nurse-love-addiction-R0047': {'A': 'Kaede laughs, says she is glad to hear that, and asks everyone to look after their health.',
                                'B': 'Nao puts her fingertip in between her lips.',
                                'C': 'Nao says that she nearly forgot to give it to Asuka, hands it over, and apologizes before leaving.',
                                'D': 'Asuka runs into her arms in tears and accepts her as an older sister.'},
 'nurse-love-addiction-R0048': {'A': 'Itsuki says that it must have been quite an ordeal for him.',
                                'B': 'Sakuya does not have any. Sakuya says stop talking about '
                                     'stupid things.'},
 'nurse-love-addiction-R0051': {'A': 'Engaged.',
                                'C': 'Recently divorced.',
                                'D': 'Married.',
                                'E': 'Single.',
                                'F': 'Dating a nurse.'},
 'nurse-love-addiction-R0053': {'A': 'A business trip.',
                                'B': 'A dispute over an exclusive dorm.',
                                'C': 'A family wedding.',
                                'E': 'A hospital emergency.',
                                'F': 'A funeral.'},
 'nurse-love-addiction-R0054': {'D': 'A dorm-room rule.', 'E': 'A disagreement over Kaede.'},
 'nurse-love-addiction-R0055': {'A': 'To celebrate the school festival.',
                                'C': 'To make Itsuki jealous.',
                                'D': 'To repay a debt.',
                                'E': "To test Asuka's feelings.",
                                'F': 'To apologize for an argument.'},
 'nurse-love-addiction-R0056': {'A': 'A love letter.',
                                'B': 'A hospital transfer request.',
                                'C': 'A signed resignation form.',
                                'D': 'A teaching contract.',
                                'F': 'A blank sheet.'},
 'nurse-love-addiction-R0057': {'A': 'Sumatriptan.',
                                'B': 'Diazepam.',
                                'C': 'Acetaminophen.',
                                'D': 'Ibuprofen.',
                                'F': 'Prednisone.'},
 'nurse-love-addiction-R0059': {'A': 'Drug-induced amnesia.',
                                'B': 'Hypnosis.',
                                'D': 'Hippocampal suppression.',
                                'E': 'Electrical stimulation.',
                                'F': 'Surgical lesioning.'},
 'nurse-love-addiction-R0061': {'A': 'Nao notices a snot bubble, tells her sister that it is unbecoming of a second-year student, and asks whether she wants to be a first-year again.',
                                'B': 'Nao understands that her sister had not eaten all day, but says that she ate far too much and asks whether she is all right.',
                                'C': 'Nao calculates that someone who died at sixty-five during her fifth-grade autumn would now be seventy-four.',
                                'D': "Nao says in any case, since they themselves haven't said "
                                     'anything, it might be best not to spread any rumors about '
                                     'it.',
                                'E': 'Nao says that the dormitory bath is too small; she appreciates the offer but only needs to warm up.'},
 'nurse-love-addiction-R0062': {'A': 'Itsuki says this is an important matter for the lab. Itsuki '
                                     'knows.',
                                'B': 'Itsuki warns that if she gets too worked up, she will wet her pants again.',
                                'C': 'Itsuki handles them independently and says that they are slightly naughty by nature.',
                                'D': 'Itsuki is surprised that Asuka would say that, then insists that it is nothing.',
                                'E': 'Itsuki says on the day, she will pretend like she has '
                                     "forgotten it's the Princess's birthday."},
 'nurse-love-addiction-R0063': {'A': 'Itsuki says morning, Princess. Itsuki is just about to head '
                                     'to bed.',
                                'B': 'Itsuki welcomes them to her secret base.',
                                'D': 'Itsuki describes thick eye shadow, baggy skirts, and a Hannya-demon brooch.',
                                'E': 'Itsuki asks Asuka what she should wear, says that she wears whatever she likes, and admits that she sometimes sleeps naked.'},
 'nurse-love-addiction-R0064': {'B': "Kaede says people's hair grows at different rates. Kaede "
                                     "says there's nothing embarrassing about an adult without "
                                     'hair in certain areas.',
                                'C': 'Kaede says that her heart is in the right place but asks her to be polite tomorrow.',
                                'E': 'Kaede says that she will distribute the hospital-training papers later so that everyone can check their details individually.'},
 'nurse-love-addiction-R0065': {'A': 'Nao says studying in a karaoke booth could make for a good '
                                     'change of pace. Nao says maybe.',
                                'D': "Nao says of course. Nao says for now, let's just do some "
                                     'window shopping and take a walk around.',
                                'E': 'Nao asks whether they have a folding umbrella because rain is forecast.'},
 'nurse-love-addiction-R0066': {'D': 'Asuka says that it is different when someone is going on a date with a lover.'},
 'nurse-love-addiction-R0067': {'A': 'Sakuya says they went swimming. Sakuya says Itsuki '
                                     'challenged her to a race to the buoy.',
                                'B': 'Sakuya says this is different. Sakuya says when they are '
                                     "surrounded by all their classmates it's obviously another "
                                     'matter.',
                                'D': "Sakuya says it's an important topic. Sakuya says clothing is "
                                     'one of the basic necessities of life.',
                                'E': 'Sakuya agrees that it is cute in its own way, then apologizes.'},
 'nurse-love-addiction-R0068': {'D': 'Asuka says that they should not share an umbrella, especially not with someone like her.'},
 'nurse-love-addiction-R0069': {'A': 'Sakuya asks whether they were talking and gaming until late the previous night.',
                                'B': "Sakuya says one wrong word and it'll sound more like "
                                     'sarcasm. Sakuya says especially to someone who wears the '
                                     'same hospital nightwear every day.',
                                'C': 'Sakuya says this is different. Sakuya says when they are '
                                     "surrounded by all their classmates it's obviously another "
                                     'matter.',
                                'D': 'Sakuya says that they have to attend because it is an important, if daunting, class.',
                                'E': 'Sakuya asks whether they plan to return to their room like that, since they cannot keep wearing wet clothes.'},
 'nurse-love-addiction-R0070': {'A': 'Asuka says they have arrived. Asuka says this is the place.',
                                'B': 'Asuka says actually, she was a little shocked.',
                                'D': "Asuka says it's not a laughing matter.",
                                'E': 'Asuka says maybe she heard it on TV.'},
 'nurse-love-addiction-R0071': {'A': 'Itsuki is just thinking about something. Itsuki says if she '
                                     'stays shut up in her room it just makes her more depressed. '
                                     "Sigh. Itsuki says she's gotten over it and now she tells her "
                                     'that she has feelings for her.',
                                'B': "Itsuki says it's about the evil life form, Virilius, which "
                                     'is eating its way through the world and the magical girl who '
                                     'fights it with the power of healing.',
                                'C': 'Itsuki says that she turned down everyone who asked to exchange forbidden items. She then suggests resting after the meal before moving on to dessert and watermelon splitting.',
                                'D': 'Itsuki says that she has moved on and now admits her feelings, then recalls turning down everyone who requested a forbidden exchange.',
                                'E': 'Itsuki suggests resting after the meal before dessert and watermelon splitting, then says that staying shut in her room would only make her more depressed.'},
 'nurse-love-addiction-R0072': {'B': 'Asuka says that it had that kind of impact, like a sudden boom.',
                                'C': 'Asuka tells Prima to stop spacing out and draw the crowd.',
                                'D': 'Asuka says that she can do it when she sets her mind to it.',
                                'E': "Asuka says nah, she couldn't do it. Asuka says a bed is no "
                                     'place to pee.'},
 'nurse-love-addiction-R0073': {'B': 'Nao tells Miss Itsuki that she should not say things like that.',
                                'D': 'Nao says aha. Nao would drop out too. Nao would chase after '
                                     'her.'},
 'nurse-love-addiction-R0074': {'B': "Itsuki says it probably wouldn't change anything."},
 'nurse-love-addiction-R0075': {'A': "Kaede says that the second-years' Capping Ceremony will be held the following week and that all first-years will attend.",
                                'C': 'Kaede says next, in November. Kaede says it is still quite a '
                                     'way ahead of them, but they will hold their annual school '
                                     'festival.',
                                'D': 'Kaede says during summer vacation, the training room will be '
                                     "available for use. Kaede says it's important to review their "
                                     'training as well.',
                                'E': 'Kaede says simple things like bed making, bathing, and blood '
                                     'pressure measurement will begin from autumn of this year.'},
 'nurse-love-addiction-R0076': {'B': 'Itsuki says that the other person is cuter than she is and more of a handful.',
                                'C': 'Itsuki says friendly competition is important. Itsuki says '
                                     'for that, rivalry is necessary.',
                                'D': 'Itsuki admits that she was probably excited and stayed up very late.',
                                'E': "Itsuki says that's true. Itsuki says fine, she will leave it to "
                                     'them then.'},
 'nurse-love-addiction-R0077': {'A': "Nao says it's not ‘standing.' It's ‘glued' onto the front "
                                     "door. Nao says it's an anatomical chart.",
                                'C': 'Nao says there are test drills from the past national exams.',
                                'D': 'Nao says that if Asuka insists on wearing her training uniform, Nao will wear hers too.',
                                'E': 'Nao says they are. Nao says actually, they are even a little '
                                     'ahead of schedule.'},
 'nurse-love-addiction-R0078': {'A': 'Nao sends her sister off, warns her to be careful, and says that the sea is dangerous at night.',
                                'C': 'Nao doubts that it will beat beach-hut fried noodles, no matter how professionally it is made.',
                                'D': "Nao wouldn't want her sister to be like her. Nao says Asuka "
                                     "is Asuka and that is what's great about her.",
                                'E': 'Nao says that she will use the leftover rice to make porridge and looks for the milk.'},
 'nurse-love-addiction-R0079': {'B': 'Asuka thanks Nao for saving her.',
                                'C': 'Asuka asks for bone-in fried chicken.',
                                'D': "Asuka agrees and says, ‘Then let's go.’",
                                'E': 'Asuka confirms that they were on the phone.'},
 'nurse-love-addiction-R0080': {'A': 'Asuka says that she does not think they can see it, even if they look very carefully.',
                                'C': 'Asuka tells Ms. Ohara that she will be fine and can get home on her own.',
                                'D': 'Asuka says miss Itsuki likes games, so she was just talking '
                                     'to her friend about them.'},
 'nurse-love-addiction-R0081': {'A': 'Asuka says that she somehow ended up like this and insists that she is not spying on them.',
                                'B': "Asuka praises Nao's perceptiveness and says that she may request Nao's ‘healing forehead’ when they get home.",
                                'D': 'Asuka tells Nao to relax and not to worry because she will be fine on her own.',
                                'E': "Asuka says Ms. Ohara didn't seem to have an umbrella with "
                                     "her. Asuka wonders if she's alright."},
 'nurse-love-addiction-R0082': {'A': 'Sakuya apologizes on behalf of both herself and Itsuki.'},
 'nurse-love-addiction-R0083': {'A': 'Kaede says the classes she will be overseeing are the '
                                     'primary subjects such as Math and English, and General '
                                     'Nursing Theory.',
                                'B': 'Kaede asks everyone to look at their papers and listen while she introduces the subjects she will oversee.',
                                'D': 'Asuka says all the subjects look so difficult.'},
 'nurse-love-addiction-R0084': {'A': 'Nao repeats that it is a lie and urges Asuka not to believe her.',
                                'B': 'Amused, Asuka remarks that the two of them really get along.',
                                'C': "Asuka says that Sakuya wants her younger sister to block Itsuki's messages."},
 'nurse-love-addiction-R0085': {'A': "Nao says it's not ‘standing.' It's ‘glued' onto the front "
                                     "door. Nao says it's an anatomical chart.",
                                'C': 'Asuka cries out that a skeleton is standing in front of the door.'},
 'nurse-love-addiction-R0086': {'A': 'Itsuki confirms that she said it was impossible for them to have ordinary fights.',
                                'B': 'Itsuki asks whether she had not been told that they never have huge fights, then distinguishes huge fights from ordinary ones.',
                                'C': "Itsuki says maybe the fact that it's autumn."},
 'nurse-love-addiction-R0087': {'A': 'Kaede says they can. Kaede is sure of it.',
                                'B': 'Asuka says that Ms. Ohara is her role model and that she wants to become a nurse just like her.'},
 'nurse-love-addiction-R0089': {'A': 'Sakuya threatens to have Itsuki arrested for violating entertainment-establishment laws.',
                                'B': 'Itsuki laughs, says that she would rather not be arrested, and withdraws the offer.',
                                'C': 'Itsuki handles them independently and says that they are slightly naughty by nature.',
                                'D': 'Sakuya says take this seriously, Itsuki. Sakuya says this '
                                     "isn't play."},
 'nurse-love-addiction-R0090': {'A': 'Asuka says that Sakuya has not returned to the dorm, although she often sees Sakuya and Itsuki talking at school.',
                                'B': "Sakuya says that's true. Sakuya supposes the ice might soon melt "
                                     'between them.'},
 'nurse-love-addiction-R0091': {'C': "Asuka says come on now, don't be shy.",
                                'D': 'Asuka says Nao, she was considering something last night '
                                     'before she slept.'},
 'nurse-love-addiction-R0092': {'A': 'Asuka tells Nao that this is not what she wanted to ask.'},
 'nurse-love-addiction-R0093': {'B': 'Itsuki says hmph. Itsuki cannot just let them leave then.',
                                'D': 'Asuka confirms that she did.'},
 'nurse-love-addiction-R0094': {'D': 'Asuka says that she is not all right, has run out of energy, and may need to replenish it.'},
 'nurse-love-addiction-R0095': {'B': 'Kaede begins morning homeroom once everyone has arrived.'},
 'nurse-love-addiction-R0096': {'B': "Nao says it's pretty humid in here since the air "
                                     "conditioning isn't working.",
                                'D': 'Itsuki says to pass her life in purity and to practice her '
                                     'profession faithfully.'},
 'nurse-love-addiction-R0097': {'A': 'Kaede reminds Asuka of her promise and says that she believed Asuka was not the kind of girl who would break it.',
                                'D': 'Kaede says that she dreamed Asuka had told the academy about her envelope.'},
 'nurse-love-addiction-R0098': {'D': 'Nao says that she is joking and only trying a new approach with her sister.'},
 'nurse-love-addiction-R0099': {'C': 'Asuka says it seems like she is taking that medicine quite '
                                     'often.'},
 'nurse-love-addiction-R0100': {'C': 'Asuka tells Nao that this is not what she wanted to ask.',
                                'F': 'Asuka protests that Nao cannot kiss her because they are sisters.'},
 'nurse-love-addiction-R0101': {'A': 'Itsuki says originally, their club was putting out an indie '
                                     'magazine focused on novels. Itsuki says it was filled with '
                                     'short stories—literary, mystery, sci-fi, all sorts.',
                                'B': 'Itsuki says that she heard there are many styles of nurse uniforms.',
                                'D': "Itsuki says the thing was, it really didn't sell very well.",
                                'E': "Itsuki says she's gotten over it and now she tells her that "
                                     'she has feelings for her.',
                                'F': 'Itsuki says —Anyway, it was just a friend. Itsuki says they '
                                     'just had a bit of an argument.'},
 'nurse-love-addiction-R0102': {'A': 'Asuka asks what Nao means by ‘the end,’ since they have not discussed anything yet.',
                                'B': 'Asuka insists that it is not that she does not want to.',
                                'D': 'Asuka wonders why Nao asked her that way, especially using the phrase ‘no matter what.’',
                                'E': 'Asuka says that it is not as though she needs to know no matter what.'},
 'nurse-love-addiction-R0103': {'A': "Nao says it's not like she does not want to do the magic "
                                     'charm or anything.',
                                'B': "Nao says she's not in her room right now. Nao says this time "
                                     "it seems like it'll be for two or three days.",
                                'C': "Nao says her mother collapsed again, so she's back at her "
                                     'home right now.',
                                'D': "Nao says if it's about Miss Sakuya, she could ask Miss "
                                     'Itsuki.',
                                'E': 'Nao heard there would be classes like this. Nao says it '
                                     'really is embarrassing.'},
 'nurse-love-addiction-R0104': {'A': "Sakuya says perhaps she could call it a ‘contract' of sorts.",
                                'B': 'Sakuya threatens to have Itsuki arrested for violating entertainment-establishment laws.',
                                'D': 'Sakuya says that the food is not well made and is too oily, just as she expected.',
                                'E': 'Sakuya says that it was written in the Teito Nursing Academy brochure and asks whether they read it.',
                                'F': 'Sakuya says that she was prepared for it, though it is still a lot.'},
 'nurse-love-addiction-R0105': {'B': 'Asuka says that she thinks she got it right this time while catching her breath.',
                                'E': 'Asuka says that all she has to do now is plug it in.',
                                'F': 'Asuka says that she can do it when she sets her mind to it.'},
 'nurse-love-addiction-R0106': {'B': 'Itsuki says that Sakuya once loved someone and wonders why that person rejected her.',
                                'E': 'Itsuki guarantees that Asuka has the potential to become a star among nerds.',
                                'F': 'Itsuki says in the past she took her to the lab, and she did '
                                     'some cosplay there.'},
 'nurse-love-addiction-R0107': {'A': 'Itsuki says there are plenty of nursing academies that '
                                     "aren't boarding schools.",
                                'B': 'Itsuki says that some stores have backup generators, though she doubts that this one does.',
                                'C': 'Itsuki says that Asuka only claims that now because she has lost, which is unlike her.',
                                'D': 'Itsuki says actually, the Prima costume looked out of place '
                                     "in the dorm. Itsuki says it's silly.",
                                'E': "Itsuki says if she was simply worried, she wouldn't leave "
                                     'her aging parent and enroll in a boarding school.',
                                'F': "Itsuki says there's no point in worrying about it now. "
                                     "Itsuki says it'll turn out however it'll turn out."},
 'nurse-love-addiction-R0108': {'A': 'Itsuki says once the pictures are ready she will give them '
                                     'to her, so look forward to it.',
                                'B': "Itsuki says she's gotten over it and now she tells her that "
                                     'she has feelings for her.',
                                'C': "Itsuki says that she and Sakuya met Little Osachi at Teito Nursing Academy's open campus before enrolling.",
                                'D': 'Itsuki says that Little Osachi approached them for a coincidental reason.',
                                'E': 'Itsuki says Sakuya dropped something, and Little Osachi '
                                     'picked it up and brought it to her.',
                                'F': 'Itsuki says that it happened last year and remarks that it has already been two years.'},
 'nurse-love-addiction-R0109': {'A': 'Sakuya says that staring silently is rude and asks what Asuka wants.',
                                'B': 'Sakuya says something about it being essential that the '
                                     'fried noodles at the beach huts contain slightly uncooked '
                                     'cabbage core.',
                                'C': 'After a silence, Sakuya agrees that they should discuss their own preferences first.',
                                'D': 'Sakuya thanks Asuka, humbly accepts the compliment, and says that Asuka looks great too.',
                                'E': 'Sakuya says Nao may have pushed her, but she was able to '
                                     'finish her homework a week ago.',
                                'F': 'Sakuya agrees that, regardless of appearances, Itsuki can behave like a rodent.'},
 'nurse-love-addiction-R0110': {'F': 'Itsuki says actually, it started over two weeks ago.'},
 'nurse-love-addiction-R0111': {'A': 'Kaede says next, in November. Kaede says it is still quite a '
                                     'way ahead of them, but they will hold their annual school '
                                     'festival.',
                                'B': 'Kaede says at the end of her vacation, there will be a '
                                     'review test for each individual subject, so make sure to '
                                     'study properly.',
                                'C': 'Kaede says in their next class, they will have to make a '
                                     'speech, so use the training room for actual practice and '
                                     'make sure that everyone has their say.',
                                'D': "Kaede says that regular classes begin after the entrance ceremony and introduces the year's curriculum.",
                                'E': 'Kaede says during summer vacation, the training room will be '
                                     "available for use. Kaede says it's important to review their "
                                     'training as well.',
                                'F': 'Kaede asks everyone to look at their papers and listen while she introduces the subjects she will oversee.'},
 'nurse-love-addiction-R0112': {'A': 'Asuka protests that they cannot do that and calls it criminal.',
                                'D': 'Asuka asks Itsuki about bondage and drugs, then asks whether hypnosis is really so easy to succumb to.',
                                'F': 'Asuka playfully tells Nao that she woke up on her own, then pretends to sleep.'},
 'nurse-love-addiction-R0113': {'A': 'Asuka asks how Itsuki and Miss Sakuya first got together.',
                                'E': 'Asuka says that she can do it when she sets her mind to it.',
                                'F': "Asuka says that's not it at all. Asuka just stammered, that's "
                                     'all.'}}


_ISSUE_PATTERNS = (
    (
        "embedded selection instruction",
        re.compile(r"\bSelect all that apply\?", re.I),
    ),
    (
        "boundary framing",
        re.compile(r"\b(?:earlier|later) boundary\b|\bboundary (?:event|events)\b", re.I),
    ),
    (
        "long-range framing",
        re.compile(
            r"\blong[- ]range\b|\bwidely separated\b|\bseparated (?:events|moments)\b|"
            r"\bcommon moments?\b.*\bacross sessions?\b",
            re.I,
        ),
    ),
    (
        "annotation thread",
        re.compile(
            r"(?:[“\"][^”\"]+[”\"]\s+thread|\b(?:storyline|narrative|music-festival)\s+thread\b)",
            re.I,
        ),
    ),
    (
        "keyword episode framing",
        re.compile(
            r"\bepisode (?:anchored by|involving|containing)\b|"
            r"\bsame episode as\b|\bepisode-mates?\b|\banchor line\b",
            re.I,
        ),
    ),
    (
        "source-material framing",
        re.compile(
            r"\bprovided dialogue\b|\bdialogue establish(?:es|ed)?\b|"
            r"\bestablished by (?:the )?dialogue turns\b|"
            r"\bgrounded in the dialogue from\b|"
            r"\bcombination of (?:the )?evidence\b|"
            r"\b(?:supplied|separated) evidence\b|"
            r"\bevidence (?:in|from) [“\"]|\bcombining the evidence from\b|"
            r"\bduring the evidence used\b|\bwhat does the evidence establish\b|"
            r"\bcommon-route passages\b",
            re.I,
        ),
    ),
    (
        "cited-material framing",
        re.compile(
            r"\bcited (?:moments?|facts?|episode)\b|\blinking multiple cited facts\b",
            re.I,
        ),
    ),
    (
        "repeated ordering template",
        re.compile(
            r"^How should these four developments be arranged in story order\? "
            r"They concern .+ and involve .+\.$",
            re.I,
        ),
    ),
    ("malformed possessive", re.compile(r"\bcharacters's\b", re.I)),
    ("mechanical clause", re.compile(r"\b(?:observes|establishes) that\b", re.I)),
    (
        "local-anchor framing",
        re.compile(
            r"\bin this exchange\b|\blocal (?:scene|event|anchor|continuation|progression)\b|"
            r"\bfollowing the local event\b|\bdescribed (?:anchor|remark)\b|"
            r"\bsame immediate exchange\b|\bbefore the scene moves on\b|"
            r"\bnearby (?:response|continuation)\b|\bevent involving nearby\b|"
            r"\blater in the same local sequence\b",
            re.I,
        ),
    ),
    (
        "route-label framing",
        re.compile(r"\broute-local\b|\broute (?:events|moments)\b|\bcommon-route relationship\b", re.I),
    ),
    (
        "abstract claim framing",
        re.compile(r"\bevent- or state-level claims\b", re.I),
    ),
    (
        "underspecified event",
        re.compile(r"^Which event involving [A-Za-z’' -]+ occurs\?$", re.I),
    ),
)


_OPTION_ISSUE_PATTERNS = (
    (
        "mechanical observation prose",
        re.compile(
            r"\b(?:[A-Z][A-Za-z’'-]* observes that|"
            r"reacts strongly and observes that)\b",
            re.I,
        ),
    ),
    (
        "mechanical consideration prose",
        re.compile(r"\b[A-Z][A-Za-z’'-]* considers (?:that|it)\b"),
    ),
    (
        "mechanical narration prose",
        re.compile(r"\bThe narration establishes that\b", re.I),
    ),
)

_LOWERCASE_OPTION_BODY_ALLOWLIST = ("eBay",)


def find_construction_issues(stem: str) -> list[str]:
    """Return benchmark-construction artifacts found in a question stem."""
    text = str(stem or "")
    return [name for name, pattern in _ISSUE_PATTERNS if pattern.search(text)]


def find_option_issues(option: str) -> list[str]:
    """Return generated reporting clauses or malformed option presentation."""
    text = str(option or "")
    issues = [
        name for name, pattern in _OPTION_ISSUE_PATTERNS if pattern.search(text)
    ]
    lowercase_match = re.match(r"^[A-Z]\.\s+([a-z]\S*)", text)
    if lowercase_match and not any(
        lowercase_match.group(1).startswith(allowed)
        for allowed in _LOWERCASE_OPTION_BODY_ALLOWLIST
    ):
        issues.append("lowercase option body")
    return issues


def make_rewrite_key(title_key: str, canonical_index: int) -> str:
    """Return the pre-publication key for a canonical QA item."""
    return f"{title_key}-R{canonical_index:04d}"


def rewrite_publication_item(
    rewrite_key: str, stem: str, options: list[str]
) -> tuple[str | None, list[str], str]:
    """Apply a pre-publication decision without mutating the source options."""
    rewritten_options = list(options)
    question_rewrite = QUESTION_REWRITES.get(rewrite_key, stem)
    if question_rewrite is None:
        return None, rewritten_options, "delete"

    option_rewrites = OPTION_REWRITES.get(rewrite_key)
    if option_rewrites:
        labelled_options: dict[str, tuple[int, str]] = {}
        for index, option in enumerate(rewritten_options):
            match = re.fullmatch(r"([A-Z])\.\s+(.+)", str(option or ""), re.S)
            if not match:
                raise ValueError(f"invalid labelled option for {rewrite_key}: {option!r}")
            letter, body = match.groups()
            if letter in labelled_options:
                raise ValueError(f"duplicate option letter for {rewrite_key}: {letter}")
            labelled_options[letter] = (index, body)

        for letter, body in option_rewrites.items():
            if not re.fullmatch(r"[A-Z]", str(letter)):
                raise ValueError(f"unexpected option rewrite letter for {rewrite_key}: {letter!r}")
            if letter not in labelled_options:
                raise ValueError(f"missing option letter for {rewrite_key}: {letter}")
            if not isinstance(body, str) or not body.strip() or body != body.strip():
                raise ValueError(f"invalid option rewrite body for {rewrite_key} {letter}")
            if re.match(r"^[A-Z]\.\s+", body):
                raise ValueError(f"labelled option rewrite body for {rewrite_key} {letter}")
            index, _old_body = labelled_options[letter]
            rewritten_options[index] = f"{letter}. {body}"

        normalized_bodies = [
            re.sub(r"\s+", " ", option.split(". ", 1)[1]).strip().casefold()
            for option in rewritten_options
        ]
        if len(normalized_bodies) != len(set(normalized_bodies)):
            raise ValueError(f"duplicate normalized option body after rewrite: {rewrite_key}")
        if len(rewritten_options) != len(options):
            raise ValueError(f"option count changed after rewrite: {rewrite_key}")

    action = (
        "rewrite"
        if rewrite_key in QUESTION_REWRITES or rewrite_key in OPTION_REWRITES
        else "unchanged"
    )
    return question_rewrite, rewritten_options, action


def rewrite_question_stem(rewrite_key: str, stem: str) -> tuple[str | None, str]:
    """Apply the explicit decision for ``rewrite_key`` to a question stem."""
    if rewrite_key not in QUESTION_REWRITES:
        return stem, "unchanged"
    rewritten_stem = QUESTION_REWRITES[rewrite_key]
    return rewritten_stem, "delete" if rewritten_stem is None else "rewrite"


__all__ = [
    "OPTION_REWRITES",
    "QUESTION_REWRITES",
    "find_construction_issues",
    "find_option_issues",
    "make_rewrite_key",
    "rewrite_publication_item",
    "rewrite_question_stem",
]
