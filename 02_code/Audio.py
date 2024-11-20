import numpy as np
import librosa
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt


class Audio:
    def __init__(self):
        self.audio_path = None
        self.audio = None

    def audio_desde_archivo(self, path):
        self.audio_path = path
        self.sr, audio = wavfile.read(path)
        if audio.ndim > 1:
            self.audio = audio[:, 0]
        else:
            self.audio = audio
        
    
    def filtrar(self, audio):
        lowcut = 250  # Frecuencia de corte inferior en Hz
        highcut = 5500  # Frecuencia de corte superior en Hz
        lowcut_normalized = lowcut / (0.5 * self.sr)
        highcut_normalized = highcut / (0.5 * self.sr)
        order = 4  # Orden del filtro
        b, a = butter(order, [lowcut_normalized, highcut_normalized], btype='band')

        if len(audio.shape) > 1:
            # Convertir a mono promediando los canales
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=float)
        filtered_audio = lfilter(b, a, audio)
        return filtered_audio
    

    def extraer_caracteristicas_audio(self, audio):
        mfcc = librosa.feature.mfcc(y = audio, sr=self.sr, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(y = audio, sr=self.sr)
        zcr = librosa.feature.zero_crossing_rate(y = audio)
        duration = librosa.get_duration(y = audio, sr = self.sr)
        # Extraer características de los primeros 100 ms del audio
        first_100ms = audio[:int(0.1 * self.sr)]
        mfcc_100ms = librosa.feature.mfcc(y=first_100ms, sr=self.sr, n_mfcc=13)
        spectral_contrast_100ms = librosa.feature.spectral_contrast(y=first_100ms, sr=self.sr)
        zcr_100ms = librosa.feature.zero_crossing_rate(y=first_100ms)

        # Concatenar las características de los primeros 100 ms con las características generales
        self.features = np.concatenate((
            mfcc.mean(axis=1), 
            spectral_contrast.mean(axis=1), 
            zcr.mean(axis=1), 
            [duration],
            mfcc_100ms.mean(axis=1),
            spectral_contrast_100ms.mean(axis=1),
            zcr_100ms.mean(axis=1)
        ))

        

    def analisis_completo(self, audio):
        audio = self.filtrar(audio)
        audio = librosa.effects.trim(audio, top_db=15, frame_length=512, hop_length=64)[0]
        audio = librosa.effects.preemphasis(audio, coef=0.95)
        audio = librosa.util.normalize(audio)
        # # Reproducir un "pip" al principio
        # duration = 0.3  # Duración del pip en segundos
        # frequency = 1000  # Frecuencia del pip en Hz
        # t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # pip = 0.5 * np.sin(2 * np.pi * frequency * t)
        # sd.play(pip, self.sr)
        # sd.wait()
        # 
        # # Reproducir el audio después de filtrar, eliminar silencios y normalizar
        # sd.play(audio, self.sr)
        # sd.wait()
        # 
        # # Reproducir un "pip" al final
        # sd.play(pip, self.sr)
        # sd.wait()
        self.extraer_caracteristicas_audio(audio)

    def mostrar_pasos_analisis_audio(self):
        audio = self.audio

        # Paso 1: Audio original
        plt.figure(figsize=(14, 5))
        plt.plot(audio)
        plt.title('Audio Original')
        plt.show()

        # Paso 2: Filtrado
        audio = self.filtrar(audio)
        plt.figure(figsize=(14, 5))
        plt.plot(audio)
        plt.title('Audio Filtrado')
        plt.show()

        # Paso 4: Trim
        audio = librosa.effects.trim(audio, top_db=15, frame_length=512, hop_length=64)[0]
        plt.figure(figsize=(14, 5))
        plt.plot(audio)
        plt.title('Audio después de Trim')
        plt.show()

        # Paso 5: Preénfasis
        audio = librosa.effects.preemphasis(audio, coef=0.95)
        plt.figure(figsize=(14, 5))
        plt.plot(audio)
        plt.title('Audio después de Preénfasis')
        plt.show()

        # Paso 6: Normalización
        audio = librosa.util.normalize(audio)
        plt.figure(figsize=(14, 5))
        plt.plot(audio)
        plt.title('Audio Normalizado')
        plt.show()

        self.extraer_caracteristicas_audio(audio)

    