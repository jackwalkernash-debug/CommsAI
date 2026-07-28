using NAudio.Wave;

namespace CommsAI;

public sealed class AudioSegmenter : IDisposable
{
    private readonly WaveFormat _inputFormat;
    private readonly string _tempFolder;
    private readonly List<byte> _buffer = new();
    private readonly Queue<byte[]> _preRoll = new();
    private readonly int _preRollChunks = 5;

    private bool _active;
    private bool _suppressed;
    private int _silentMilliseconds;
    private int _totalMilliseconds;
    private int _voicedMilliseconds;
    private double _peakLevel;

    public double StartThreshold { get; set; } = 0.020;
    public double StopThreshold { get; set; } = 0.010;
    public double MinimumPeakLevel { get; set; } = 0.030;
    public int SilenceToFinishMilliseconds { get; set; } = 650;
    public int MinimumSpeechMilliseconds { get; set; } = 500;
    public int MinimumVoicedMilliseconds { get; set; } = 300;
    public int MaximumSegmentMilliseconds { get; set; } = 9000;

    public event Action<string>? SegmentReady;
    public event Action<double>? LevelMeasured;

    public bool Suppressed
    {
        get => _suppressed;
        set
        {
            if (_suppressed == value)
                return;

            _suppressed = value;
            Reset();
        }
    }

    public AudioSegmenter(WaveFormat inputFormat, string tempFolder)
    {
        _inputFormat = inputFormat;
        _tempFolder = tempFolder;
        Directory.CreateDirectory(_tempFolder);
    }

    public void Push(byte[] data, int count)
    {
        if (_suppressed)
        {
            LevelMeasured?.Invoke(0);
            return;
        }

        var copy = new byte[count];
        Buffer.BlockCopy(data, 0, copy, 0, count);

        var level = CalculateRms(copy, count, _inputFormat);
        LevelMeasured?.Invoke(level);

        var milliseconds = Math.Max(
            1,
            (int)Math.Round(
                count * 1000d / _inputFormat.AverageBytesPerSecond
            )
        );

        _preRoll.Enqueue(copy);
        while (_preRoll.Count > _preRollChunks)
            _preRoll.Dequeue();

        if (!_active)
        {
            if (level >= StartThreshold)
            {
                _active = true;
                _buffer.Clear();
                foreach (var chunk in _preRoll)
                    _buffer.AddRange(chunk);

                _totalMilliseconds = milliseconds * _preRoll.Count;
                _silentMilliseconds = 0;
                _voicedMilliseconds = milliseconds;
                _peakLevel = level;
            }

            return;
        }

        _buffer.AddRange(copy);
        _totalMilliseconds += milliseconds;
        _peakLevel = Math.Max(_peakLevel, level);
        if (level >= StopThreshold)
            _voicedMilliseconds += milliseconds;

        _silentMilliseconds = level < StopThreshold
            ? _silentMilliseconds + milliseconds
            : 0;

        var shouldFinish =
            _silentMilliseconds >= SilenceToFinishMilliseconds ||
            _totalMilliseconds >= MaximumSegmentMilliseconds;

        if (!shouldFinish)
            return;

        var meaningfulMilliseconds =
            _totalMilliseconds - _silentMilliseconds;

        if (meaningfulMilliseconds >= MinimumSpeechMilliseconds &&
            _voicedMilliseconds >= MinimumVoicedMilliseconds &&
            _peakLevel >= MinimumPeakLevel)
            SaveSegment();

        Reset();
    }

    private void SaveSegment()
    {
        var filename = Path.Combine(
            _tempFolder,
            $"segment_{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}.wav"
        );

        using var writer = new WaveFileWriter(filename, _inputFormat);
        var bytes = _buffer.ToArray();
        writer.Write(bytes, 0, bytes.Length);

        SegmentReady?.Invoke(filename);
    }

    private void Reset()
    {
        _active = false;
        _buffer.Clear();
        _preRoll.Clear();
        _silentMilliseconds = 0;
        _totalMilliseconds = 0;
        _voicedMilliseconds = 0;
        _peakLevel = 0;
    }

    private static double CalculateRms(
        byte[] buffer,
        int count,
        WaveFormat format
    )
    {
        double sum = 0;
        long samples = 0;

        if (format.Encoding == WaveFormatEncoding.IeeeFloat &&
            format.BitsPerSample == 32)
        {
            for (var i = 0; i + 3 < count; i += 4)
            {
                var sample = BitConverter.ToSingle(buffer, i);
                sum += sample * sample;
                samples++;
            }
        }
        else if (format.BitsPerSample == 16)
        {
            for (var i = 0; i + 1 < count; i += 2)
            {
                var sample = BitConverter.ToInt16(buffer, i) / 32768d;
                sum += sample * sample;
                samples++;
            }
        }

        return samples == 0 ? 0 : Math.Sqrt(sum / samples);
    }

    public void Dispose()
    {
        Reset();
    }
}
