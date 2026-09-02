"""Manually approved natural-language rewrites for publication QA stems."""

from __future__ import annotations

import re


QUESTION_REWRITES: dict[str, str | None] = {
    "fault-milestone-two-R0011": (
        "What happens as Selphine returns to herself after the confrontation? "
        "Select all that apply?"
    ),
    "fault-milestone-two-R0028": (
        "How does Sol's relationship with Selphine's group change over the course "
        "of the story? Select all that apply?"
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
        "What does Selphine's behavior reveal about her struggle to preserve her own "
        "identity against Queen Rhegan's influence? Select all that apply?"
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
        "What does the story reveal about how the Path-down affects identity and "
        "behavior? Select all that apply?"
    ),
    "fault-milestone-two-R0026": (
        "What does Selphine say to Sol when they first meet in Neo Sasary?"
    ),
    "fault-milestone-two-R0031": (
        "In what order do these events involving the group's sea voyage occur?"
    ),
    "fault-milestone-two-R0034": (
        "Why is Selphine afraid of inheriting Queen Rhegan's identity? Select all "
        "that apply?"
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
        "How does Sol respond to his theft from Mil and the responsibility he later "
        "assumes? Select all that apply?"
    ),
    "fault-milestone-two-R0056": (
        "How does Selphine turn her distrust of outsiders into an investigation "
        "centered on Sol? Select all that apply?"
    ),
    "fault-milestone-two-R0057": (
        "In what order do the group's visit to the bathhouse and Sol's theft there "
        "unfold?"
    ),
    "fault-milestone-two-R0059": (
        "What happens when Ritona searches the bathhouse for a missing child? Select "
        "all that apply?"
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
        "How does Sceatoire respond after hearing Rune's name? Select all that apply?"
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
        "How does the group's trust in Greus change after he makes Ritona's survival "
        "conditional on taking him to Rughzenhaide? Select all that apply?"
    ),
    "fault-milestone-two-R0092": (
        "In what order does the Vita Domain facility become involved in Ritona's "
        "treatment?"
    ),
    "fault-milestone-two-R0094": (
        "How do Ritona and Selphine disagree about whether Greus has broken his "
        "promise? Select all that apply?"
    ),
    "fault-milestone-two-R0099": (
        "What role does the Vita Domain play in Ritona's care, from its first mention "
        "through her promised release? Select all that apply?"
    ),
    "fault-milestone-two-R0100": (
        "In what order do Selphine and Mil's experiences with cooking occur?"
    ),
    "fault-milestone-two-R0103": (
        "What does the story reveal about sagiolla's role in Mil's illness and the "
        "pharmacist's exploitation? Select all that apply?"
    ),
    "fault-milestone-two-R0105": (
        "In what order does the group's judgment of Greus change?"
    ),
    "fault-milestone-two-R0109": (
        "How does Sol's responsibility for Mil change after his theft and violent "
        "confrontation with the pharmacist? Select all that apply?"
    ),
    "fault-milestone-two-R0113": (
        "In what order does the story reveal how sagiolla is prepared, used, grown, "
        "and exposed as ineffective?"
    ),
    "heart-of-the-woods-R0005": (
        "How does Madison's view of her future with Taranormal and Tara change? "
        "Select all that apply?"
    ),
    "heart-of-the-woods-R0006": (
        "How does the story gradually reveal Geladura's true identity and role? "
        "Select all that apply?"
    ),
    "heart-of-the-woods-R0007": (
        "How do the supernatural events in Eysenfeld change what Tara and Madison "
        "can prove? Select all that apply?"
    ),
    "heart-of-the-woods-R0008": (
        "How do Tara and Madison repair their friendship while Madison's future with "
        "Taranormal remains unsettled? Select all that apply?"
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
        "What do Amber and Marina learn when they find the stranded car and receive "
        "the treasure journal? Select all that apply?"
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
        "How does Amber's fatigue affect her and Marina's travel plans? Select all "
        "that apply?"
    ),
    "highway-blossoms-R0046": (
        "What do Amber, Joseph, and Mariah reveal while discussing Canyon de Chelly? "
        "Select all that apply?"
    ),
    "highway-blossoms-R0048": (
        "In what order does Amber's fatigue affect the journey and lead her to rest?"
    ),
    "highway-blossoms-R0049": (
        "By that scene, has Amber told Marina that she can see through Marina's "
        "cheerful tone and fears losing her?"
    ),
    "highway-blossoms-R0052": (
        "What happens during Amber and Marina's uncomfortable rest-stop encounter "
        "with the trucker? Select all that apply?"
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
        "What happens when Amber and Marina unexpectedly meet Joseph in town? Select "
        "all that apply?"
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
        "What do Amber and Marina conclude while looking for Angel's Landing? Select "
        "all that apply?"
    ),
    "highway-blossoms-R0068": (
        "At this point, has Amber told Marina that her disappointment at Angel's "
        "Landing is tied to Gramps' plans and her growing feelings for Marina?"
    ),
    "highway-blossoms-R0074": (
        "What do Amber and Marina reveal while discussing past relationships and "
        "their own feelings? Select all that apply?"
    ),
    "highway-blossoms-R0079": (
        "In what order do Amber and Marina discuss their dating histories?"
    ),
    "highway-blossoms-R0080": (
        "What does Amber say about her nausea and the street artist? Select all that "
        "apply?"
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
        "What do the characters reveal while playing blackjack and reflecting on their "
        "trip? Select all that apply?"
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
        "What do Amber and Marina say about Amber's unusual breakfast? Select all "
        "that apply?"
    ),
    "highway-blossoms-R0095": (
        "What happens when Amber prepares to risk the motorhome and Mariah intervenes? "
        "Select all that apply?"
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
        "What do Amber and Marina say while trying a festival drink and waiting for "
        "conditions to improve? Select all that apply?"
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
        "How does Asuka's motivation to become a nurse develop from her school career "
        "survey to her admiration for Ms. Ohara? Select all that apply?"
    ),
    "nurse-love-addiction-R0019": (
        "Which statements occur after Itsuki notices that Asuka has many questions "
        "but before Itsuki says Asuka's submission angers her? Select all that apply?"
    ),
    "nurse-love-addiction-R0020": (
        "How do Asuka's attempts to become independent from Nao reveal the sisters' "
        "continuing dependence on each other? Select all that apply?"
    ),
    "nurse-love-addiction-R0021": (
        "Which statements occur after Asuka dismisses the situation as a dream but "
        "before Nao groans while Asuka examines the medicine bottle? Select all that "
        "apply?"
    ),
    "nurse-love-addiction-R0022": (
        "How does Nao's ‘first and last date’ change Asuka and Nao's understanding of "
        "their bond? Select all that apply?"
    ),
    "nurse-love-addiction-R0023": (
        "Which statements occur after Nao suggests treating Asuka properly with "
        "medicine but before Nao recalls Open Campus Day? Select all that apply?"
    ),
    "nurse-love-addiction-R0024": (
        "Which statements occur after Itsuki describes falling out of bed but before "
        "Asuka asks whether she mentioned those things? Select all that apply?"
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
        "Which of Asuka's other thoughts and statements occur while she asks Itsuki "
        "whether she is meeting Miss Takeda? Select all that apply?"
    ),
    "nurse-love-addiction-R0101": (
        "What else does Itsuki say while explaining why her club switched from a "
        "literary magazine to magical-girl material? Select all that apply?"
    ),
    "nurse-love-addiction-R0102": (
        "What else does Asuka say or think during her conversation with Nao about "
        "umbrellas? Select all that apply?"
    ),
    "nurse-love-addiction-R0103": (
        "What else does Nao say while warning Asuka that she ate too much after "
        "fasting for a day? Select all that apply?"
    ),
    "nurse-love-addiction-R0104": (
        "What else does Sakuya say during the conversation in which she threatens to "
        "hit Itsuki until she takes back what she said? Select all that apply?"
    ),
    "nurse-love-addiction-R0105": (
        "What else does Asuka say or think when she denies that she is turning to "
        "religion? Select all that apply?"
    ),
    "nurse-love-addiction-R0106": (
        "What does Itsuki say while describing the club's cosplay activities and "
        "Asuka's potential appeal to nerds? Select all that apply?"
    ),
    "nurse-love-addiction-R0107": (
        "What does Itsuki say while explaining why Sakuya's boarding-school choice "
        "shows more than simple concern for her mother? Select all that apply?"
    ),
    "nurse-love-addiction-R0109": (
        "What else does Sakuya say while joining the students' lively beach "
        "conversation? Select all that apply?"
    ),
    "nurse-love-addiction-R0110": (
        "What else does Itsuki say while inviting Asuka and Nao to the seaside? "
        "Select all that apply?"
    ),
    "nurse-love-addiction-R0111": (
        "What other study instructions does Kaede give while explaining the purpose "
        "of the vacation review tests? Select all that apply?"
    ),
    "nurse-love-addiction-R0112": (
        "What else does Asuka say or think during the conversation in which she "
        "insists that she really wants to change? Select all that apply?"
    ),
    "nurse-love-addiction-R0113": (
        "What else does Asuka say or think while discussing how Itsuki and Sakuya got "
        "together? Select all that apply?"
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


_ISSUE_PATTERNS = (
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


def find_construction_issues(stem: str) -> list[str]:
    """Return benchmark-construction artifacts found in a question stem."""
    text = str(stem or "")
    return [name for name, pattern in _ISSUE_PATTERNS if pattern.search(text)]


def make_rewrite_key(title_key: str, canonical_index: int) -> str:
    """Return the pre-publication key for a canonical QA item."""
    return f"{title_key}-R{canonical_index:04d}"


def rewrite_publication_item(
    rewrite_key: str, stem: str, options: list[str]
) -> tuple[str | None, list[str], str]:
    """Apply a pre-publication decision without mutating the source options."""
    rewritten_options = list(options)
    if rewrite_key not in QUESTION_REWRITES:
        return stem, rewritten_options, "unchanged"
    rewrite = QUESTION_REWRITES[rewrite_key]
    if rewrite is None:
        return None, rewritten_options, "delete"
    return rewrite, rewritten_options, "rewrite"


def rewrite_question_stem(rewrite_key: str, stem: str) -> tuple[str | None, str]:
    """Apply the explicit decision for ``rewrite_key`` to a question stem."""
    rewritten_stem, _options, action = rewrite_publication_item(rewrite_key, stem, [])
    return rewritten_stem, action


__all__ = [
    "QUESTION_REWRITES",
    "find_construction_issues",
    "make_rewrite_key",
    "rewrite_publication_item",
    "rewrite_question_stem",
]
