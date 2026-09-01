"""Manually approved natural-language rewrites for publication QA stems."""

from __future__ import annotations

import re


QUESTION_REWRITES: dict[str, str | None] = {
    "fault-milestone-two-Q0011": (
        "What happens as Selphine returns to herself after the confrontation? "
        "Select all that apply."
    ),
    "fault-milestone-two-Q0028": (
        "How does Sol's relationship with Selphine's group change over the course "
        "of the story? Select all that apply."
    ),
    "highway-blossoms-Q0013": (
        "In what order do Amber and Marina discuss their plans to attend the music "
        "festival?"
    ),
    "nurse-love-addiction-Q0001": (
        "After the teacher's words resound through the classroom but before Itsuki "
        "says she has to leave for a while, what does Asuka do?"
    ),
    "nurse-love-addiction-Q0066": (
        "What does Asuka say after Sakuya tells her to stop talking gibberish and "
        "answer the question?"
    ),
    "nurse-love-addiction-Q0108": (
        "Which of Itsuki's other statements occur in the scene where she tells "
        "Asuka, ‘Tell me the truth and I’ll get really mad at you’?"
    ),
    "a-kiss-for-the-petals-Q0004": (
        "Which of Miya's family problems, if any, has Risa already learned about "
        "when she sarcastically calls Miya's parents ‘great’?"
    ),
    "a-kiss-for-the-petals-Q0006": (
        "What ultimately happens after Miya proposes that she and Risa attend the "
        "overnight study camp together?"
    ),
    "fault-milestone-two-Q0010": (
        "How do Selphine and Ritona respond once Selphine returns to herself after "
        "the confrontation?"
    ),
    "fault-milestone-two-Q0014": (
        "What does Selphine's behavior reveal about her struggle to preserve her own "
        "identity against Queen Rhegan's influence? Select all that apply."
    ),
    "fault-milestone-two-Q0015": (
        "In what order is Selphine cautioned about battlekravte, asked about a new "
        "sensation, discussed as someone's ward, and identified as her real self?"
    ),
    "fault-milestone-two-Q0017": (
        "In what order does the story reveal these facts about the Path-down?"
    ),
    "fault-milestone-two-Q0019": (
        "In what order do these encounters and discussions involving Melano occur?"
    ),
    "fault-milestone-two-Q0020": (
        "What does the story reveal about how the Path-down affects identity and "
        "behavior? Select all that apply."
    ),
    "fault-milestone-two-Q0026": (
        "What does Selphine say to Sol when they first meet in Neo Sasary?"
    ),
    "fault-milestone-two-Q0031": (
        "In what order do these events involving the group's sea voyage occur?"
    ),
    "fault-milestone-two-Q0034": (
        "Why is Selphine afraid of inheriting Queen Rhegan's identity? Select all "
        "that apply."
    ),
    "fault-milestone-two-Q0036": (
        "In what order is Sol rebuked, pursued, brought into the house, and entrusted "
        "with Mil's care?"
    ),
    "fault-milestone-two-Q0038": (
        "In what order does the story introduce these uses of mana and manakravte?"
    ),
    "fault-milestone-two-Q0048": (
        "In what order are sediment stones acquired, used on the field, and "
        "recognized later in the story?"
    ),
    "fault-milestone-two-Q0050": (
        "In what order does Selphine confront Queen Rhegan's influence on her "
        "identity?"
    ),
    "fault-milestone-two-Q0052": (
        "What does Ritona ask people at the bathhouse while searching for the "
        "missing child?"
    ),
    "fault-milestone-two-Q0054": (
        "How does Sol respond to his theft from Mil and the responsibility he later "
        "assumes? Select all that apply."
    ),
    "fault-milestone-two-Q0056": (
        "How does Selphine turn her distrust of outsiders into an investigation "
        "centered on Sol? Select all that apply."
    ),
    "fault-milestone-two-Q0057": (
        "In what order do the group's visit to the bathhouse and Sol's theft there "
        "unfold?"
    ),
    "fault-milestone-two-Q0059": (
        "What happens when Ritona searches the bathhouse for a missing child? Select "
        "all that apply."
    ),
    "fault-milestone-two-Q0068": (
        "In what order does Sol's relationship with Mil and the group change?"
    ),
    "fault-milestone-two-Q0072": (
        "In what order does the story introduce, sell, and revisit the fertile soil?"
    ),
    "fault-milestone-two-Q0073": (
        "What formal diagnosis, if any, does Greus give Mil when Selphine asks him "
        "to examine her?"
    ),
    "fault-milestone-two-Q0078": (
        "What does Sceatoire claim happened between her and someone named Rune?"
    ),
    "fault-milestone-two-Q0079": (
        "How does Sceatoire respond after hearing Rune's name? Select all that apply."
    ),
    "fault-milestone-two-Q0082": (
        "What dosage schedule, if any, does Rupika give for Ritona's medicine while "
        "explaining that only a professional can administer her care?"
    ),
    "fault-milestone-two-Q0084": (
        "In what order do scenes show Rune tracking an enemy, discussing seafood, "
        "displaying her language skills, and sharing food with Selphine?"
    ),
    "fault-milestone-two-Q0085": (
        "How does the group's trust in Greus change after he makes Ritona's survival "
        "conditional on taking him to Rughzenhaide? Select all that apply."
    ),
    "fault-milestone-two-Q0092": (
        "In what order does the Vita Domain facility become involved in Ritona's "
        "treatment?"
    ),
    "fault-milestone-two-Q0094": (
        "How do Ritona and Selphine disagree about whether Greus has broken his "
        "promise? Select all that apply."
    ),
    "fault-milestone-two-Q0099": (
        "What role does the Vita Domain play in Ritona's care, from its first mention "
        "through her promised release? Select all that apply."
    ),
    "fault-milestone-two-Q0100": (
        "In what order do Selphine and Mil's experiences with cooking occur?"
    ),
    "fault-milestone-two-Q0103": (
        "What does the story reveal about sagiolla's role in Mil's illness and the "
        "pharmacist's exploitation? Select all that apply."
    ),
    "fault-milestone-two-Q0105": (
        "In what order does the group's judgment of Greus change?"
    ),
    "fault-milestone-two-Q0109": (
        "How does Sol's responsibility for Mil change after his theft and violent "
        "confrontation with the pharmacist? Select all that apply."
    ),
    "fault-milestone-two-Q0113": (
        "In what order does the story reveal how sagiolla is prepared, used, grown, "
        "and exposed as ineffective?"
    ),
    "heart-of-the-woods-Q0005": (
        "How does Madison's view of her future with Taranormal and Tara change? "
        "Select all that apply."
    ),
    "heart-of-the-woods-Q0006": (
        "How does the story gradually reveal Geladura's true identity and role? "
        "Select all that apply."
    ),
    "heart-of-the-woods-Q0007": (
        "How do the supernatural events in Eysenfeld change what Tara and Madison "
        "can prove? Select all that apply."
    ),
    "heart-of-the-woods-Q0008": (
        "How do Tara and Madison repair their friendship while Madison's future with "
        "Taranormal remains unsettled? Select all that apply."
    ),
    "heart-of-the-woods-Q0023": (
        "In what order does Abigail's transition from ghostly existence to ordinary "
        "human sensation unfold?"
    ),
    "heart-of-the-woods-Q0026": (
        "In what order does Madison move from refusing the fairy-queen role to "
        "accepting the crown?"
    ),
    "heart-of-the-woods-Q0029": (
        "In what order do these milestones in Madison and Abigail's relationship "
        "occur?"
    ),
    "heart-of-the-woods-Q0030": (
        "In what order do Morgan's warnings about Evelyn and Madison's final response "
        "unfold?"
    ),
    "heart-of-the-woods-Q0034": (
        "In what order do Tara and Madison's expectations about finding supernatural "
        "proof change?"
    ),
    "heart-of-the-woods-Q0037": (
        "In what order do these milestones in Tara and Morgan's relationship occur?"
    ),
    "heart-of-the-woods-Q0045": (
        "In what order does Abigail begin to understand and imagine life in the "
        "modern world?"
    ),
    "highway-blossoms-Q0002": (
        "What does Marina suggest doing while the group is sightseeing before dark?"
    ),
    "highway-blossoms-Q0005": (
        "Does Amber ever tell Mariah that she sees her as a reckless person she can "
        "vent to but not trust?"
    ),
    "highway-blossoms-Q0006": (
        "Does Amber tell Marina that their reunion at the festival makes her happy "
        "and reminds her how much she loves about Marina?"
    ),
    "highway-blossoms-Q0007": (
        "Does Amber tell the stranded girl that she thinks the girl's old car is in "
        "poor condition?"
    ),
    "highway-blossoms-Q0008": (
        "What does Amber do after encountering Marina stranded by the roadside?"
    ),
    "highway-blossoms-Q0012": (
        "Does Amber tell Marina that she swerved because she regretted becoming too "
        "comfortable and saying too much?"
    ),
    "highway-blossoms-Q0015": (
        "What do Amber and Marina learn when they find the stranded car and receive "
        "the treasure journal? Select all that apply."
    ),
    "highway-blossoms-Q0017": (
        "Does Amber tell Marina that the journal entry leaves her stumped while "
        "Marina is using the payphone?"
    ),
    "highway-blossoms-Q0018": (
        "How do Amber and BandanaGuy respond while arguing about who contributed to "
        "the search? Select all that apply."
    ),
    "highway-blossoms-Q0020": (
        "How does Joseph respond after his group contributes to the trouble with "
        "Marina's car?"
    ),
    "highway-blossoms-Q0024": (
        "In what order do Amber and Marina discover what happened to Marina's car and "
        "receive Joseph's offer of help?"
    ),
    "highway-blossoms-Q0025": (
        "Does Amber admit to Marina that the journal's ‘split path’ and ‘detour’ clues "
        "overwhelm her?"
    ),
    "highway-blossoms-Q0028": (
        "Does Amber tell Marina that she finds Marina's constant smile cute and is "
        "afraid of her growing feelings?"
    ),
    "highway-blossoms-Q0035": (
        "Does Amber tell Marina that Marina's praise makes her feel she cannot live "
        "up to Marina's opinion of her?"
    ),
    "highway-blossoms-Q0040": (
        "In what order do Amber and Marina take risks while searching the ruins?"
    ),
    "highway-blossoms-Q0044": (
        "How does Amber's fatigue affect her and Marina's travel plans? Select all "
        "that apply."
    ),
    "highway-blossoms-Q0046": (
        "What do Amber, Joseph, and Mariah reveal while discussing Canyon de Chelly? "
        "Select all that apply."
    ),
    "highway-blossoms-Q0048": (
        "In what order does Amber's fatigue affect the journey and lead her to rest?"
    ),
    "highway-blossoms-Q0049": (
        "Does Amber tell Marina that she can see through Marina's cheerful tone and "
        "fears losing her?"
    ),
    "highway-blossoms-Q0052": (
        "What happens during Amber and Marina's uncomfortable rest-stop encounter "
        "with the trucker? Select all that apply."
    ),
    "highway-blossoms-Q0053": (
        "Does Amber tell Marina that she is attracted to her while Marina rests with "
        "her feet on the dashboard?"
    ),
    "highway-blossoms-Q0054": (
        "In what order do Amber and Marina encounter potentially dangerous strangers "
        "during the treasure hunt?"
    ),
    "highway-blossoms-Q0059": (
        "In what order do Amber and Marina's conversations about chocolate and "
        "getting food occur?"
    ),
    "highway-blossoms-Q0060": (
        "What happens when Amber and Marina unexpectedly meet Joseph in town? Select "
        "all that apply."
    ),
    "highway-blossoms-Q0063": (
        "What attitude does Marina express when she tells Amber, ‘Don't worry, I "
        "believe! Sounds awesome’?"
    ),
    "highway-blossoms-Q0065": (
        "Does Amber tell Marina that she is worried about the police while they chase "
        "Mariah's motorhome?"
    ),
    "highway-blossoms-Q0066": (
        "What do Amber and Marina conclude while looking for Angel's Landing? Select "
        "all that apply."
    ),
    "highway-blossoms-Q0068": (
        "Does Amber tell Marina that her disappointment at Angel's Landing is tied "
        "to Gramps' plans and her growing feelings for Marina?"
    ),
    "highway-blossoms-Q0074": (
        "What do Amber and Marina reveal while discussing past relationships and "
        "their own feelings? Select all that apply."
    ),
    "highway-blossoms-Q0079": (
        "In what order do Amber and Marina discuss their dating histories?"
    ),
    "highway-blossoms-Q0080": (
        "What does Amber say about her nausea and the street artist? Select all that "
        "apply."
    ),
    "highway-blossoms-Q0082": (
        "Does Amber tell Marina about her guilt over not wanting to get over Gramps "
        "while she waits for Marina to return?"
    ),
    "highway-blossoms-Q0086": (
        "Does Amber tell Marina that looking forward to their outing also makes her "
        "feel she is doing something wrong?"
    ),
    "highway-blossoms-Q0087": (
        "What do the characters reveal while playing blackjack and reflecting on their "
        "trip? Select all that apply."
    ),
    "highway-blossoms-Q0090": (
        "Does Amber tell Marina the depth of her guilt over Gramps and how important "
        "Marina has become to her after the diner argument?"
    ),
    "highway-blossoms-Q0092": (
        "In what order does Amber respond after learning Marina has lost her share of "
        "the treasure in Vegas?"
    ),
    "highway-blossoms-Q0093": (
        "What do Amber and Marina say about Amber's unusual breakfast? Select all "
        "that apply."
    ),
    "highway-blossoms-Q0095": (
        "What happens when Amber prepares to risk the motorhome and Mariah intervenes? "
        "Select all that apply."
    ),
    "highway-blossoms-Q0102": (
        "In what order does Amber decide to send Marina home after losing the "
        "treasure?"
    ),
    "highway-blossoms-Q0103": (
        "In what order does Amber's plan to attend the music festival progress from "
        "her first explanation to their arrival?"
    ),
    "highway-blossoms-Q0105": (
        "What do Amber and Marina say while trying a festival drink and waiting for "
        "conditions to improve? Select all that apply."
    ),
    "nurse-love-addiction-Q0002": (
        "After Asuka sleepily boasts that she woke up on her own but before she "
        "recalls General Nursing Theory, what happens when Itsuki leaves?"
    ),
    "nurse-love-addiction-Q0003": (
        "After Ms. Ohara smiles while explaining but before Asuka recognizes her "
        "voice, what does Itsuki say?"
    ),
    "nurse-love-addiction-Q0004": (
        "After Asuka's attempt at independence collapses but before Kaede warns her "
        "about the doll's weight, what does Asuka notice outside the dorm?"
    ),
    "nurse-love-addiction-Q0005": (
        "After Nao stares blankly at Asuka but before Itsuki explains the limits of "
        "Sakuya's power, what does Asuka say they need to do?"
    ),
    "nurse-love-addiction-Q0006": (
        "After Asuka gets into bed that night but before Ms. Ohara reacts to the "
        "tension in the room, what message from Itsuki does Nao read?"
    ),
    "nurse-love-addiction-Q0007": (
        "After Nao complains that the alarm cannot wake Asuka but before Asuka makes "
        "her inner declaration, what does Itsuki say about such a power?"
    ),
    "nurse-love-addiction-Q0008": (
        "After Asuka says she slept instead of training but before someone sees her "
        "cosplay outfit, what childcare instruction does Kaede give?"
    ),
    "nurse-love-addiction-Q0009": (
        "After Asuka asks whether the Osachi family took her in but before calling "
        "Nao her true sister, what does Asuka say?"
    ),
    "nurse-love-addiction-Q0010": (
        "After Sakuya says they could not find an endless summer but before Nao "
        "protests that they are not blood sisters, what does Nao say?"
    ),
    "nurse-love-addiction-Q0011": (
        "After Asuka cries that she cannot pronounce the term but before she lies "
        "alone in her quiet room, what happens in the classroom?"
    ),
    "nurse-love-addiction-Q0012": (
        "After Asuka's attempt to get Nao's attention fails but before Sakuya thanks "
        "Nao for bringing them, what does Asuka think might already be closed?"
    ),
    "nurse-love-addiction-Q0013": (
        "After Asuka wonders why she is pretending to sleep but before Nao recalls "
        "Asuka's rainy-season hospital stay, what is Nao preparing for dinner?"
    ),
    "nurse-love-addiction-Q0014": (
        "After Nao suggests doing homework but before Asuka recalls Itsuki saying "
        "something was easy to memorize, what does Asuka envy?"
    ),
    "nurse-love-addiction-Q0015": (
        "After Asuka meets Machi and Michi outside the academy but before Nao's tone "
        "turns mischievous, what does Asuka do to coax her?"
    ),
    "nurse-love-addiction-Q0016": (
        "After Kaede lists the classes she will oversee but before Nao says there is "
        "no hurry to decide, whom does Asuka ask Nao about?"
    ),
    "nurse-love-addiction-Q0017": (
        "After Ms. Ohara gently explains the upcoming events but before Itsuki recites "
        "the pledge, how does the class react?"
    ),
    "nurse-love-addiction-Q0018": (
        "How does Asuka's motivation to become a nurse develop from her school career "
        "survey to her admiration for Ms. Ohara? Select all that apply."
    ),
    "nurse-love-addiction-Q0019": (
        "Which statements occur after Itsuki notices that Asuka has many questions "
        "but before Itsuki says Asuka's submission angers her? Select all that apply."
    ),
    "nurse-love-addiction-Q0020": (
        "How do Asuka's attempts to become independent from Nao reveal the sisters' "
        "continuing dependence on each other? Select all that apply."
    ),
    "nurse-love-addiction-Q0021": (
        "Which statements occur after Asuka dismisses the situation as a dream but "
        "before Nao groans while Asuka examines the medicine bottle? Select all that "
        "apply."
    ),
    "nurse-love-addiction-Q0022": (
        "How does Nao's ‘first and last date’ change Asuka and Nao's understanding of "
        "their bond? Select all that apply."
    ),
    "nurse-love-addiction-Q0023": (
        "Which statements occur after Nao suggests treating Asuka properly with "
        "medicine but before Nao recalls Open Campus Day? Select all that apply."
    ),
    "nurse-love-addiction-Q0024": (
        "Which statements occur after Itsuki describes falling out of bed but before "
        "Asuka asks whether she mentioned those things? Select all that apply."
    ),
    "nurse-love-addiction-Q0056": (
        "What, if anything, does Asuka learn about the contents of Kaede's resignation "
        "letter when she first sees the envelope?"
    ),
    "nurse-love-addiction-Q0060": (
        "Who wrote The Girl Who Chased Stars, according to Asuka's recollection of "
        "the childhood picture book?"
    ),
    "nurse-love-addiction-Q0061": (
        "What does Nao ask Asuka after Asuka privately thinks, ‘But, then again, "
        "maybe she does’?"
    ),
    "nurse-love-addiction-Q0062": (
        "How does Itsuki respond when Asuka asks why she is making a silly face?"
    ),
    "nurse-love-addiction-Q0063": (
        "What details does Itsuki point out in the photograph after Asuka claims she "
        "was a bad girl in junior high?"
    ),
    "nurse-love-addiction-Q0064": (
        "What does Kaede say about hospital training after Asuka realizes she will "
        "visit Yuki's hospital again?"
    ),
    "nurse-love-addiction-Q0065": (
        "How does Nao answer when Asuka asks if she is some kind of witch?"
    ),
    "nurse-love-addiction-Q0067": (
        "How does Sakuya respond when Itsuki notes that she has been hospitalized "
        "before?"
    ),
    "nurse-love-addiction-Q0068": (
        "What does Asuka say after Itsuki asks Sakuya whether she wants another "
        "shower?"
    ),
    "nurse-love-addiction-Q0069": (
        "What does Sakuya ask after Asuka compliments her skin?"
    ),
    "nurse-love-addiction-Q0070": (
        "How does Asuka respond when Itsuki ends her story with ‘And then we ended up "
        "dating’?"
    ),
    "nurse-love-addiction-Q0071": (
        "How does Itsuki describe the story after Asuka asks why she is dressed up?"
    ),
    "nurse-love-addiction-Q0072": (
        "What does Asuka say after Itsuki promises that her answer will not be a lie?"
    ),
    "nurse-love-addiction-Q0073": (
        "How does Nao respond when Asuka says she does not think she will run away?"
    ),
    "nurse-love-addiction-Q0074": (
        "What does Itsuki say after Asuka admits that Itsuki's warning scared her?"
    ),
    "nurse-love-addiction-Q0075": (
        "What does Kaede announce after Asuka says she will send a text when she gets "
        "home?"
    ),
    "nurse-love-addiction-Q0076": (
        "What does Itsuki say after Sakuya suggests going to the nurse station to "
        "introduce themselves?"
    ),
    "nurse-love-addiction-Q0077": (
        "What does Nao say after Asuka remarks that studying at karaoke is equally "
        "strange?"
    ),
    "nurse-love-addiction-Q0078": (
        "How does Nao respond when Itsuki praises her honesty in contrast with Asuka?"
    ),
    "nurse-love-addiction-Q0079": (
        "How does Asuka respond when Nao reminds her of ‘the other side of the world’?"
    ),
    "nurse-love-addiction-Q0080": (
        "What does Asuka tell Ms. Ohara after Kaede says people's hair grows at "
        "different rates?"
    ),
    "nurse-love-addiction-Q0081": (
        "How does Asuka respond when Nao says that Asuka's rainy-season headaches "
        "are her only recurring health problem?"
    ),
    "nurse-love-addiction-Q0082": (
        "In what order do Sakuya and Asuka discuss Sakuya's apology and making up "
        "Asuka's missed studies?"
    ),
    "nurse-love-addiction-Q0083": (
        "In what order do Kaede and Asuka discuss the classes Kaede will oversee?"
    ),
    "nurse-love-addiction-Q0084": (
        "In what order do Asuka and Nao read Itsuki and Sakuya's contradictory text "
        "messages?"
    ),
    "nurse-love-addiction-Q0085": (
        "In what order do Nao and Asuka discuss a gift before Asuka mistakes an "
        "anatomical chart for a skeleton?"
    ),
    "nurse-love-addiction-Q0086": (
        "In what order do Itsuki and Asuka argue about Itsuki and Sakuya's fights?"
    ),
    "nurse-love-addiction-Q0087": (
        "In what order do Kaede and Asuka discuss Asuka's chances of winning the "
        "Nightingale Award?"
    ),
    "nurse-love-addiction-Q0088": (
        "In what order do Asuka and Nao imagine relaxing on a porch with a cat and "
        "rice crackers?"
    ),
    "nurse-love-addiction-Q0089": (
        "In what order do Itsuki and Sakuya joke and argue about Itsuki's questionable "
        "work?"
    ),
    "nurse-love-addiction-Q0090": (
        "In what order do Asuka and Sakuya discuss Sakuya's improving relationship "
        "with Itsuki?"
    ),
    "nurse-love-addiction-Q0091": (
        "In what order do Asuka and Nao discuss breakfast and Asuka's plan to become "
        "more independent?"
    ),
    "nurse-love-addiction-Q0092": (
        "In what order does Asuka ask Itsuki about the kiss she witnessed?"
    ),
    "nurse-love-addiction-Q0093": (
        "In what order does Itsuki confront Asuka about overhearing her phone call?"
    ),
    "nurse-love-addiction-Q0094": (
        "In what order does Asuka tease Nao by asking for another dose of "
        "‘replenishment’?"
    ),
    "nurse-love-addiction-Q0095": (
        "In what order do Itsuki's arrival and the start of morning homeroom unfold?"
    ),
    "nurse-love-addiction-Q0096": (
        "In what order do these events surrounding the Nightingale pledge ceremony "
        "occur?"
    ),
    "nurse-love-addiction-Q0097": (
        "In what order does Kaede recount her nightmare about Asuka revealing the "
        "resignation letter?"
    ),
    "nurse-love-addiction-Q0098": (
        "In what order does Asuka think through kissing Nao before Nao reveals that "
        "she was joking?"
    ),
    "nurse-love-addiction-Q0100": (
        "Which of Asuka's other thoughts and statements occur while she asks Itsuki "
        "whether she is meeting Miss Takeda? Select all that apply."
    ),
    "nurse-love-addiction-Q0101": (
        "What else does Itsuki say while explaining why her club switched from a "
        "literary magazine to magical-girl material? Select all that apply."
    ),
    "nurse-love-addiction-Q0102": (
        "What else does Asuka say or think during her conversation with Nao about "
        "umbrellas? Select all that apply."
    ),
    "nurse-love-addiction-Q0103": (
        "What else does Nao say while warning Asuka that she ate too much after "
        "fasting for a day? Select all that apply."
    ),
    "nurse-love-addiction-Q0104": (
        "What else does Sakuya say during the conversation in which she threatens to "
        "hit Itsuki until she takes back what she said? Select all that apply."
    ),
    "nurse-love-addiction-Q0105": (
        "What else does Asuka say or think when she denies that she is turning to "
        "religion? Select all that apply."
    ),
    "nurse-love-addiction-Q0106": (
        "What does Itsuki say while describing the club's cosplay activities and "
        "Asuka's potential appeal to nerds? Select all that apply."
    ),
    "nurse-love-addiction-Q0107": (
        "What does Itsuki say while explaining why Sakuya's boarding-school choice "
        "shows more than simple concern for her mother? Select all that apply."
    ),
    "nurse-love-addiction-Q0109": (
        "What else does Sakuya say while joining the students' lively beach "
        "conversation? Select all that apply."
    ),
    "nurse-love-addiction-Q0110": (
        "What else does Itsuki say while inviting Asuka and Nao to the seaside? "
        "Select all that apply."
    ),
    "nurse-love-addiction-Q0111": (
        "What other study instructions does Kaede give while explaining the purpose "
        "of the vacation review tests? Select all that apply."
    ),
    "nurse-love-addiction-Q0112": (
        "What else does Asuka say or think during the conversation in which she "
        "insists that she really wants to change? Select all that apply."
    ),
    "nurse-love-addiction-Q0113": (
        "What else does Asuka say or think while discussing how Itsuki and Sakuya got "
        "together? Select all that apply."
    ),
}


_ISSUE_PATTERNS = (
    (
        "boundary framing",
        re.compile(r"\b(?:earlier|later) boundary\b|\bboundary (?:event|events)\b", re.I),
    ),
    (
        "long-range framing",
        re.compile(r"\blong[- ]range\b|\bwidely separated\b|\bseparated (?:events|moments)\b", re.I),
    ),
    ("annotation thread", re.compile(r"\bthread\b", re.I)),
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
    ("malformed possessive", re.compile(r"\bcharacters's\b", re.I)),
    ("mechanical clause", re.compile(r"\b(?:observes|establishes) that\b", re.I)),
    (
        "local-anchor framing",
        re.compile(
            r"\bin this exchange\b|\blocal (?:scene|event|anchor|continuation)\b|"
            r"\bfollowing the local event\b|\bdescribed (?:anchor|remark)\b|"
            r"\bsame immediate exchange\b|\bbefore the scene moves on\b|"
            r"\bnearby (?:response|continuation)\b|\bevent involving nearby\b",
            re.I,
        ),
    ),
    ("route-label framing", re.compile(r"\broute-local\b|\broute (?:events|moments)\b", re.I)),
    (
        "abstract claim framing",
        re.compile(r"\bevent- or state-level claims\b", re.I),
    ),
    (
        "underspecified event",
        re.compile(r"^Which event involving [A-Za-z’' -]+ occurs\?$", re.I),
    ),
)


def find_construction_issues(stem: str) -> list[str]:
    """Return benchmark-construction artifacts found in a question stem."""
    text = str(stem or "")
    return [name for name, pattern in _ISSUE_PATTERNS if pattern.search(text)]


def rewrite_question_stem(qa_id: str, stem: str) -> tuple[str | None, str]:
    """Apply the explicit decision for ``qa_id`` without changing unlisted stems."""
    if qa_id not in QUESTION_REWRITES:
        return stem, "unchanged"
    rewrite = QUESTION_REWRITES[qa_id]
    if rewrite is None:
        return None, "delete"
    return rewrite, "rewrite"


__all__ = [
    "QUESTION_REWRITES",
    "find_construction_issues",
    "rewrite_question_stem",
]
