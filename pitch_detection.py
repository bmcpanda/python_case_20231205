import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def fundamental_frequency_pitch_track(file_path, file_name):
    y, sr = librosa.load(file_path + file_name)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6'))
    times = librosa.times_like(f0)

    # 设置 y 轴刻度为C3到C6的音符范围
    notes = librosa.core.midi_to_note(
        np.arange(librosa.note_to_midi('C3'), librosa.note_to_midi('C6') + 1))
    notes_hz = librosa.core.note_to_hz(notes)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        D, x_axis='time', y_axis='fft_note', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.yticks(notes_hz, notes)
    plt.ylim([librosa.note_to_hz('C3'), librosa.note_to_hz('C6')])
    plt.savefig('./static/' + file_name + '.jpg')
    # plt.show()


if __name__ == '__main__':
    file_path = ('../../game/')
    file_name = ('1_sanpo_(Vocals).mp3')
    fundamental_frequency_pitch_track(file_path, file_name)
