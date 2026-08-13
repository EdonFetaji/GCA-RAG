from extractor_agent.extractor_agent import run_extractor_pipeline
import json
document_text = """
Meridian Robotics announced on March 14, 2024 that it has acquired
Northfield Automation, a smaller competitor based in Austin, Texas.
The deal was confirmed by Meridian Robotics CEO Elena Voss during a
press conference at the company's headquarters in Seattle.

Voss said the acquisition would accelerate development of the
company's flagship product, the AtlasArm industrial robot, which
Meridian Robotics first launched in 2021. Northfield Automation's
founder, Marcus Chen, will join Meridian Robotics as Vice President
of Engineering.

The Global Robotics Alliance, an industry trade group, released a
statement praising the merger as a sign of consolidation in the
robotics sector. Analysts expect the combined company to compete
more directly with Sento Dynamics, a rival headquartered in Boston.

Meridian Robotics plans to close the acquisition by June 2024,
pending regulatory approval.
"""
kg = run_extractor_pipeline(document_text)
print(kg)
json.dump(kg, open("data/extractor_agent_sample.json", "w"), indent=2)