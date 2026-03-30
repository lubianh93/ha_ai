"""Edge TTS voice definitions for AI Hub integration.

This file contains the complete list of Edge TTS voices.
Separated from const.py to keep that file manageable.
"""

from typing import Final

# Edge TTS Voices - Complete list
# Maps voice ID to language code
EDGE_TTS_VOICES: Final = {
    # Chinese (Simplified)
    'zh-CN-XiaoxiaoNeural': 'zh-CN',
    'zh-CN-XiaoyiNeural': 'zh-CN',
    'zh-CN-YunjianNeural': 'zh-CN',
    'zh-CN-YunxiNeural': 'zh-CN',
    'zh-CN-YunxiaNeural': 'zh-CN',
    'zh-CN-YunyangNeural': 'zh-CN',
    # Chinese (Hong Kong)
    'zh-HK-HiuGaaiNeural': 'zh-HK',
    'zh-HK-HiuMaanNeural': 'zh-HK',
    'zh-HK-WanLungNeural': 'zh-HK',
    # Chinese (Taiwan)
    'zh-TW-HsiaoChenNeural': 'zh-TW',
    'zh-TW-YunJheNeural': 'zh-TW',
    'zh-TW-HsiaoYuNeural': 'zh-TW',
    # English (US)
    'en-US-AvaNeural': 'en-US',
    'en-US-AndrewNeural': 'en-US',
    'en-US-EmmaNeural': 'en-US',
    'en-US-BrianNeural': 'en-US',
    'en-US-JennyNeural': 'en-US',
    'en-US-GuyNeural': 'en-US',
    'en-US-AriaNeural': 'en-US',
    'en-US-DavisNeural': 'en-US',
    'en-US-AmberNeural': 'en-US',
    'en-US-AnaNeural': 'en-US',
    'en-US-AshleyNeural': 'en-US',
    'en-US-BrandonNeural': 'en-US',
    'en-US-ChristopherNeural': 'en-US',
    'en-US-CoraNeural': 'en-US',
    'en-US-ElizabethNeural': 'en-US',
    'en-US-EricNeural': 'en-US',
    'en-US-JacobNeural': 'en-US',
    'en-US-MichelleNeural': 'en-US',
    'en-US-MonicaNeural': 'en-US',
    'en-US-RogerNeural': 'en-US',
    'en-US-SaraNeural': 'en-US',
    'en-US-SteffanNeural': 'en-US',
    # English (UK)
    'en-GB-LibbyNeural': 'en-GB',
    'en-GB-MaisieNeural': 'en-GB',
    'en-GB-RyanNeural': 'en-GB',
    'en-GB-SoniaNeural': 'en-GB',
    'en-GB-ThomasNeural': 'en-GB',
    # English (Australia)
    'en-AU-NatashaNeural': 'en-AU',
    'en-AU-WilliamNeural': 'en-AU',
    # English (Canada)
    'en-CA-ClaraNeural': 'en-CA',
    'en-CA-LiamNeural': 'en-CA',
    # English (Other)
    'en-HK-SamNeural': 'en-HK',
    'en-HK-YanNeural': 'en-HK',
    'en-IE-ConnorNeural': 'en-IE',
    'en-IE-EmilyNeural': 'en-IE',
    'en-IN-NeerjaNeural': 'en-IN',
    'en-IN-PrabhatNeural': 'en-IN',
    'en-NZ-MitchellNeural': 'en-NZ',
    'en-NZ-MollyNeural': 'en-NZ',
    'en-SG-LunaNeural': 'en-SG',
    'en-SG-WayneNeural': 'en-SG',
    'en-ZA-LeahNeural': 'en-ZA',
    'en-ZA-LukeNeural': 'en-ZA',
    # Japanese
    'ja-JP-KeitaNeural': 'ja-JP',
    'ja-JP-NanamiNeural': 'ja-JP',
    # Korean
    'ko-KR-InJoonNeural': 'ko-KR',
    'ko-KR-SunHiNeural': 'ko-KR',
    # French
    'fr-FR-DeniseNeural': 'fr-FR',
    'fr-FR-EloiseNeural': 'fr-FR',
    'fr-FR-HenriNeural': 'fr-FR',
    'fr-CA-AntoineNeural': 'fr-CA',
    'fr-CA-JeanNeural': 'fr-CA',
    'fr-CA-SylvieNeural': 'fr-CA',
    # German
    'de-DE-AmalaNeural': 'de-DE',
    'de-DE-ConradNeural': 'de-DE',
    'de-DE-KatjaNeural': 'de-DE',
    'de-DE-KillianNeural': 'de-DE',
    'de-AT-IngridNeural': 'de-AT',
    'de-AT-JonasNeural': 'de-AT',
    'de-CH-JanNeural': 'de-CH',
    'de-CH-LeniNeural': 'de-CH',
    # Spanish
    'es-ES-AlvaroNeural': 'es-ES',
    'es-ES-ElviraNeural': 'es-ES',
    'es-MX-DaliaNeural': 'es-MX',
    'es-MX-JorgeNeural': 'es-MX',
    'es-AR-ElenaNeural': 'es-AR',
    'es-AR-TomasNeural': 'es-AR',
    'es-CO-GonzaloNeural': 'es-CO',
    'es-CO-SalomeNeural': 'es-CO',
    'es-US-AlonsoNeural': 'es-US',
    'es-US-PalomaNeural': 'es-US',
    # Italian
    'it-IT-DiegoNeural': 'it-IT',
    'it-IT-ElsaNeural': 'it-IT',
    'it-IT-IsabellaNeural': 'it-IT',
    # Portuguese
    'pt-BR-AntonioNeural': 'pt-BR',
    'pt-BR-FranciscaNeural': 'pt-BR',
    'pt-PT-DuarteNeural': 'pt-PT',
    'pt-PT-RaquelNeural': 'pt-PT',
    # Russian
    'ru-RU-DmitryNeural': 'ru-RU',
    'ru-RU-SvetlanaNeural': 'ru-RU',
    # Arabic
    'ar-SA-HamedNeural': 'ar-SA',
    'ar-SA-ZariyahNeural': 'ar-SA',
    'ar-EG-SalmaNeural': 'ar-EG',
    'ar-EG-ShakirNeural': 'ar-EG',
    # Hindi
    'hi-IN-MadhurNeural': 'hi-IN',
    'hi-IN-SwaraNeural': 'hi-IN',
    # Dutch
    'nl-NL-ColetteNeural': 'nl-NL',
    'nl-NL-FennaNeural': 'nl-NL',
    'nl-NL-MaartenNeural': 'nl-NL',
    'nl-BE-ArnaudNeural': 'nl-BE',
    'nl-BE-DenaNeural': 'nl-BE',
    # Polish
    'pl-PL-MarekNeural': 'pl-PL',
    'pl-PL-ZofiaNeural': 'pl-PL',
    # Turkish
    'tr-TR-AhmetNeural': 'tr-TR',
    'tr-TR-EmelNeural': 'tr-TR',
    # Vietnamese
    'vi-VN-HoaiMyNeural': 'vi-VN',
    'vi-VN-NamMinhNeural': 'vi-VN',
    # Thai
    'th-TH-NiwatNeural': 'th-TH',
    'th-TH-PremwadeeNeural': 'th-TH',
    # Indonesian
    'id-ID-ArdiNeural': 'id-ID',
    'id-ID-GadisNeural': 'id-ID',
    # Swedish
    'sv-SE-MattiasNeural': 'sv-SE',
    'sv-SE-SofieNeural': 'sv-SE',
    # Norwegian
    'nb-NO-FinnNeural': 'nb-NO',
    'nb-NO-PernilleNeural': 'nb-NO',
    # Danish
    'da-DK-ChristelNeural': 'da-DK',
    'da-DK-JeppeNeural': 'da-DK',
    # Finnish
    'fi-FI-HarriNeural': 'fi-FI',
    'fi-FI-NooraNeural': 'fi-FI',
    # Greek
    'el-GR-AthinaNeural': 'el-GR',
    'el-GR-NestorasNeural': 'el-GR',
    # Czech
    'cs-CZ-AntoninNeural': 'cs-CZ',
    'cs-CZ-VlastaNeural': 'cs-CZ',
    # Romanian
    'ro-RO-AlinaNeural': 'ro-RO',
    'ro-RO-EmilNeural': 'ro-RO',
    # Hungarian
    'hu-HU-NoemiNeural': 'hu-HU',
    'hu-HU-TamasNeural': 'hu-HU',
    # Ukrainian
    'uk-UA-OstapNeural': 'uk-UA',
    'uk-UA-PolinaNeural': 'uk-UA',
    # Hebrew
    'he-IL-AvriNeural': 'he-IL',
    'he-IL-HilaNeural': 'he-IL',
    # Malay
    'ms-MY-OsmanNeural': 'ms-MY',
    'ms-MY-YasminNeural': 'ms-MY',
    # Filipino
    'fil-PH-AngeloNeural': 'fil-PH',
    'fil-PH-BlessicaNeural': 'fil-PH',
}
