"""
scripts/create_sample_pdfs.py
Generate realistic multi-page ISRO mission PDFs for demo/testing.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
title_style   = ParagraphStyle("Title2",   parent=styles["Title"],   fontSize=18, spaceAfter=12, alignment=TA_CENTER)
h1_style      = ParagraphStyle("H1",       parent=styles["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6)
h2_style      = ParagraphStyle("H2",       parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
body_style    = ParagraphStyle("Body",     parent=styles["Normal"],   fontSize=10, spaceAfter=8,  leading=14, alignment=TA_JUSTIFY)
caption_style = ParagraphStyle("Caption",  parent=styles["Normal"],   fontSize=9,  textColor=(0.4,0.4,0.4), spaceAfter=6)

def build(filename: str, story: list) -> None:
    path = OUT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    doc.build(story)
    print(f"  Created: {path}")

# ── Chandrayaan-3 ─────────────────────────────────────────────────────────────
def chandrayaan3():
    s = []
    s += [Paragraph("Chandrayaan-3 Mission Report", title_style),
          Paragraph("Indian Space Research Organisation (ISRO) — 2023", caption_style),
          Spacer(1, 0.4*cm)]

    s += [Paragraph("1. Mission Overview", h1_style),
          Paragraph(
            "Chandrayaan-3 is India's third lunar exploration mission and the successor to "
            "Chandrayaan-2. The mission objective is to demonstrate safe and soft landing on "
            "the lunar surface, rover roving on the Moon, and in-situ scientific experiments. "
            "The mission was launched on 14 July 2023 aboard the LVM3-M4 launch vehicle from "
            "Satish Dhawan Space Centre (SDSC) SHAR, Sriharikota. The spacecraft successfully "
            "achieved a soft landing near the lunar south pole on 23 August 2023, making India "
            "the fourth country to achieve a soft lunar landing and the first to land near the "
            "south pole.", body_style),
          Paragraph(
            "The total mission cost was approximately INR 615 crore (approximately USD 75 million), "
            "making Chandrayaan-3 one of the most cost-effective lunar missions in history. "
            "The mission demonstrated ISRO's capability for interplanetary exploration and "
            "advanced autonomous navigation.", body_style)]

    s += [Paragraph("2. Mission Architecture", h1_style),
          Paragraph("2.1 Propulsion Module", h2_style),
          Paragraph(
            "The Chandrayaan-3 propulsion module carries the spacecraft from launch injection "
            "to lunar orbit. It uses a 440 N liquid apogee motor (LAM) as the main engine for "
            "orbit-raising manoeuvres and lunar orbit insertion. The propulsion module has a "
            "total mass of 2148 kg including 1696.39 kg of propellant. Four 50 N thrusters "
            "provide attitude control. The propulsion module also carries the SHAPE "
            "(Spectro-polarimetry of HAbitable Planet Earth) scientific payload, which studies "
            "Earth's spectral and polarimetric signatures from lunar orbit.", body_style),
          Paragraph("2.2 Lander Module (Vikram)", h2_style),
          Paragraph(
            "The lander module, named Vikram, has a mass of 1752 kg including the Pragyan rover. "
            "It is equipped with four throttleable engines of 800 N thrust each for the powered "
            "descent phase. The lander uses a Lander Hazard Detection and Avoidance Camera "
            "(LHDAC) and a Laser Doppler Velocimeter (LDV) for autonomous hazard detection "
            "during the final descent. Solar panels generate 738 W of power. The lander "
            "communicates with IDSN (Indian Deep Space Network) at Byalalu via S-band and "
            "X-band links, and also uses the Chandrayaan-2 orbiter as a communication relay.", body_style),
          PageBreak()]

    s += [Paragraph("2.3 Rover (Pragyan)", h1_style),
          Paragraph(
            "The Pragyan rover has a mass of 26 kg and is designed for one lunar day "
            "(approximately 14 Earth days) of operation. It moves at a speed of 1 cm/s on "
            "the lunar surface. The rover carries two scientific payloads: APXS (Alpha Particle "
            "X-ray Spectrometer) for elemental analysis of the lunar surface, and LIBS (Laser "
            "Induced Breakdown Spectroscope) for determining the elemental composition of rocks "
            "and soil. The rover generates 50 W of power from its solar panel.", body_style),
          Paragraph("3. Landing Site", h1_style),
          Paragraph(
            "The Chandrayaan-3 lander targeted a landing site near the lunar south pole at "
            "approximately 69.37 degrees South latitude and 32.35 degrees East longitude. "
            "The south polar region is scientifically significant because permanently shadowed "
            "craters in this region may contain water ice deposits. These water ice deposits "
            "could serve as resources for future lunar exploration and potential human habitation "
            "on the Moon. The landing site was selected based on orbital imagery from "
            "Chandrayaan-2 and other lunar orbiters.", body_style),
          Paragraph("4. Scientific Discoveries", h1_style),
          Paragraph(
            "The Pragyan rover confirmed the presence of sulphur on the lunar south pole surface "
            "for the first time through in-situ measurements using the LIBS instrument. "
            "The APXS instrument detected aluminium, calcium, iron, chromium, titanium, "
            "manganese, silicon, and oxygen in the lunar soil. The ChaSTE (Chandra's Surface "
            "Thermophysical Experiment) instrument on the lander measured the temperature "
            "profile of the lunar surface and found a significant temperature difference of "
            "approximately 60 degrees Celsius between the surface and 8 cm below the surface.", body_style),
          Paragraph("5. Communication System", h1_style),
          Paragraph(
            "Chandrayaan-3 uses a dual-frequency communication system. The lander communicates "
            "directly with the Indian Deep Space Network (IDSN) ground station at Byalalu, "
            "Bengaluru using S-band (uplink) and X-band (downlink) frequencies. Additionally, "
            "the Chandrayaan-2 orbiter, which remains operational in lunar orbit, serves as a "
            "communication relay between the lander and Earth. This relay capability was "
            "pre-planned as a contingency and was successfully demonstrated during the mission.", body_style)]

    build("chandrayaan3_mission_report.pdf", s)


# ── Mangalyaan ────────────────────────────────────────────────────────────────
def mangalyaan():
    s = []
    s += [Paragraph("Mars Orbiter Mission (Mangalyaan)", title_style),
          Paragraph("Indian Space Research Organisation (ISRO) — 2013–2022", caption_style),
          Spacer(1, 0.4*cm)]

    s += [Paragraph("1. Mission Overview", h1_style),
          Paragraph(
            "The Mars Orbiter Mission (MOM), informally known as Mangalyaan, is India's first "
            "interplanetary mission. It was launched on 5 November 2013 by the Polar Satellite "
            "Launch Vehicle (PSLV-C25) from Satish Dhawan Space Centre, Sriharikota. The "
            "spacecraft was inserted into Mars orbit on 24 September 2014, making India the "
            "first Asian nation to reach Martian orbit and the first nation in the world to "
            "do so in its maiden attempt. The total mission cost was approximately INR 450 crore "
            "(approximately USD 74 million at the time), making it the least expensive Mars "
            "mission to date.", body_style),
          Paragraph("2. Mission Objectives", h1_style),
          Paragraph(
            "The primary objective of Mangalyaan was to develop the technologies required for "
            "design, planning, management, and operations of an interplanetary mission. The "
            "secondary scientific objectives included exploration of Mars surface features, "
            "morphology, mineralogy, and the Martian atmosphere using indigenous scientific "
            "instruments. The mission demonstrated ISRO's capability to develop and operate "
            "a spacecraft over interplanetary distances.", body_style),
          PageBreak()]

    s += [Paragraph("3. Scientific Instruments", h1_style),
          Paragraph(
            "Mangalyaan carries five scientific instruments with a total payload mass of "
            "15 kg. The instruments are: LAP (Lyman Alpha Photometer) for measuring the "
            "relative abundance of deuterium and hydrogen in the Martian upper atmosphere; "
            "MSM (Methane Sensor for Mars) for measuring methane in the Martian atmosphere; "
            "TIS (Thermal Infrared Imaging Spectrometer) for mapping surface composition and "
            "mineralogy; MCC (Mars Colour Camera) for taking images of the Martian surface "
            "and its moons Phobos and Deimos; and MENCA (Mars Exospheric Neutral Composition "
            "Analyser) for studying the neutral composition of the Martian exosphere.", body_style),
          Paragraph("4. Spacecraft and Orbit", h1_style),
          Paragraph(
            "The Mangalyaan spacecraft has a launch mass of 1337 kg including 852 kg of "
            "propellant. It uses a 440 N liquid apogee motor for major manoeuvres. The "
            "spacecraft is in a highly elliptical orbit around Mars with a periapsis of "
            "approximately 421 km and an apoapsis of approximately 76,993 km, with an "
            "orbital period of about 72.7 hours. The orbit was chosen to allow observation "
            "of the entire Martian disc and to study the upper atmosphere.", body_style),
          Paragraph("5. Journey to Mars", h1_style),
          Paragraph(
            "After launch on 5 November 2013, Mangalyaan performed a series of orbit-raising "
            "manoeuvres around Earth before the Trans-Mars Injection (TMI) burn on "
            "1 December 2013. The spacecraft travelled approximately 780 million km over "
            "298 days to reach Mars. The Mars Orbit Insertion (MOI) burn was performed on "
            "24 September 2014, lasting approximately 24 minutes, reducing the spacecraft "
            "velocity sufficiently to be captured by Martian gravity. The mission operated "
            "for over 8 years before communication was lost in September 2022.", body_style)]

    build("mangalyaan_mission_report.pdf", s)


# ── Aditya-L1 ─────────────────────────────────────────────────────────────────
def aditya_l1():
    s = []
    s += [Paragraph("Aditya-L1 Solar Observatory Mission", title_style),
          Paragraph("Indian Space Research Organisation (ISRO) — 2023", caption_style),
          Spacer(1, 0.4*cm)]

    s += [Paragraph("1. Mission Overview", h1_style),
          Paragraph(
            "Aditya-L1 is India's first dedicated solar observatory mission. The spacecraft "
            "was launched on 2 September 2023 by the PSLV-C57 rocket from Satish Dhawan Space "
            "Centre, Sriharikota. The mission is designed to study the Sun from a halo orbit "
            "around the Sun-Earth Lagrange point 1 (L1), located approximately 1.5 million km "
            "from Earth. The L1 point provides a unique vantage point for continuous, "
            "uninterrupted observation of the Sun without any occultation or eclipse.", body_style),
          Paragraph("2. Scientific Objectives", h1_style),
          Paragraph(
            "The primary scientific objectives of Aditya-L1 are to study the solar corona and "
            "its heating mechanism, solar wind acceleration, coupling and dynamics of the solar "
            "atmosphere, solar wind distribution and temperature anisotropy, and the origin of "
            "coronal mass ejections (CMEs) and flares. Understanding these phenomena is crucial "
            "for space weather prediction, which affects satellite operations, power grids, and "
            "communication systems on Earth.", body_style),
          PageBreak()]

    s += [Paragraph("3. Scientific Payloads", h1_style),
          Paragraph(
            "Aditya-L1 carries seven scientific payloads with a total mass of approximately "
            "244 kg. The payloads are: VELC (Visible Emission Line Coronagraph) — the primary "
            "payload, designed to study the solar corona and dynamics of coronal mass ejections, "
            "built by the Indian Institute of Astrophysics; SUIT (Solar Ultraviolet Imaging "
            "Telescope) for imaging the solar photosphere and chromosphere in near-UV; "
            "SoLEXS (Solar Low Energy X-ray Spectrometer) for monitoring solar X-ray flares; "
            "HEL1OS (High Energy L1 Orbiting X-ray Spectrometer) for studying hard X-ray "
            "flares; ASPEX (Aditya Solar wind Particle EXperiment) for studying solar wind "
            "protons and alpha particles; PAPA (Plasma Analyser Package for Aditya) for "
            "measuring solar wind electrons and heavy ions; and MAG (Advanced Tri-axial "
            "High Resolution Digital Magnetometers) for measuring the interplanetary magnetic "
            "field.", body_style),
          Paragraph("4. The L1 Lagrange Point", h1_style),
          Paragraph(
            "The Sun-Earth Lagrange point 1 (L1) is a gravitational equilibrium point located "
            "approximately 1.5 million km from Earth in the direction of the Sun, which is "
            "about 1% of the Earth-Sun distance. A spacecraft at L1 requires minimal fuel to "
            "maintain its position. The key advantage of the L1 point for solar observation "
            "is that it allows continuous, uninterrupted viewing of the Sun without any "
            "occultation or eclipse caused by Earth or Moon. Aditya-L1 reached its halo orbit "
            "around L1 on 6 January 2024.", body_style),
          Paragraph("5. VELC Instrument", h1_style),
          Paragraph(
            "The Visible Emission Line Coronagraph (VELC) is the largest and most complex "
            "payload on Aditya-L1. It is designed to image the solar corona continuously and "
            "study the dynamics of coronal mass ejections (CMEs). VELC can simultaneously "
            "image the corona in three broadband channels and perform spectroscopy in one "
            "channel. It can observe the corona from 1.05 to 3 solar radii. The instrument "
            "is expected to transmit approximately 1440 images per day to the ground station.", body_style)]

    build("aditya_l1_mission_report.pdf", s)


# ── Gaganyaan ─────────────────────────────────────────────────────────────────
def gaganyaan():
    s = []
    s += [Paragraph("Gaganyaan Human Spaceflight Programme", title_style),
          Paragraph("Indian Space Research Organisation (ISRO) — Programme Overview", caption_style),
          Spacer(1, 0.4*cm)]

    s += [Paragraph("1. Programme Overview", h1_style),
          Paragraph(
            "Gaganyaan is India's first human spaceflight programme, aimed at demonstrating "
            "the capability to send Indian astronauts (Vyomanauts) to low Earth orbit and "
            "return them safely to Earth. The programme objective is to launch a crew of "
            "three astronauts to an orbit of 400 km altitude for a mission duration of "
            "3 days before returning safely to Earth with a splashdown in the Bay of Bengal. "
            "The programme will make India the fourth country in the world to independently "
            "send humans to space, after the USA, Russia, and China.", body_style),
          Paragraph("2. Launch Vehicle", h1_style),
          Paragraph(
            "Gaganyaan will use the Human Rated LVM3 (HLVM3) launch vehicle, which is a "
            "modified version of the LVM3 (formerly GSLV Mk III) rocket. The HLVM3 has been "
            "upgraded with additional safety features including a Crew Escape System (CES) "
            "that can pull the crew module away from the rocket in case of an emergency during "
            "launch or ascent. The rocket uses two solid strap-on boosters (S200), a liquid "
            "core stage (L110) with two Vikas engines, and a cryogenic upper stage (C25) with "
            "a CE-20 engine.", body_style),
          PageBreak()]

    s += [Paragraph("3. Crew Module", h1_style),
          Paragraph(
            "The Gaganyaan Crew Module (CM) is a pressurised module designed to carry three "
            "astronauts. It has a mass of approximately 5700 kg and an inner volume of "
            "approximately 8 cubic metres. The module is equipped with life support systems "
            "that maintain a shirt-sleeve environment for the crew. The Crew Module is designed "
            "to withstand re-entry heating and splashes down in the sea, where it is recovered "
            "by the Indian Navy. The module uses a parachute system for deceleration during "
            "re-entry.", body_style),
          Paragraph("4. Astronaut Training", h1_style),
          Paragraph(
            "Four Indian Air Force pilots have been selected as astronaut candidates for the "
            "Gaganyaan programme. They underwent initial training in Russia at the Yuri Gagarin "
            "Cosmonaut Training Centre (GCTC) in Star City. Training includes physical "
            "conditioning, spacecraft systems familiarisation, survival training, and "
            "microgravity adaptation. Additional mission-specific training is conducted at "
            "ISRO's astronaut training facility in Bengaluru.", body_style),
          Paragraph("5. Mission Sequence", h1_style),
          Paragraph(
            "The Gaganyaan mission sequence begins with launch from Satish Dhawan Space Centre "
            "at Sriharikota. After reaching the target orbit of 400 km, the crew will conduct "
            "scientific experiments and technology demonstrations over 3 days. Re-entry is "
            "initiated by a deorbit burn, followed by atmospheric re-entry with peak heating "
            "of approximately 1600 degrees Celsius on the heat shield. The crew module "
            "deploys parachutes at an altitude of approximately 15 km and splashes down in "
            "the Bay of Bengal, approximately 500 km off the coast of India.", body_style)]

    build("gaganyaan_mission_report.pdf", s)


if __name__ == "__main__":
    print("Generating sample ISRO mission PDFs …")
    chandrayaan3()
    mangalyaan()
    aditya_l1()
    gaganyaan()
    print("Done. PDFs saved to data/raw/")
