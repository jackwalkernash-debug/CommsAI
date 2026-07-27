using System.Speech.Synthesis;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace CommsAI;

public sealed class MainForm : Form
{
    private readonly ComboBox _deviceBox = new();
    private readonly ComboBox _modelBox = new();
    private readonly ComboBox _computeBox = new();
    private readonly CheckBox _speakBox = new();
    private readonly Button _refreshButton = new();
    private readonly Button _startButton = new();
    private readonly Button _stopButton = new();
    private readonly ProgressBar _meter = new();
    private readonly Label _status = new();
    private readonly Label _language = new();
    private readonly Label _processing = new();
    private readonly TextBox _original = new();
    private readonly TextBox _english = new();
    private readonly DataGridView _history = new();

    private MMDeviceEnumerator? _enumerator;
    private WasapiLoopbackCapture? _capture;
    private AudioSegmenter? _segmenter;
    private BackendClient? _backend;
    private SpeechSynthesizer? _speech;
    private readonly Queue<string> _pendingSegments = new();
    private bool _backendReady;
    private bool _processingSegment;

    public MainForm()
    {
        Text = "CommsAI";
        Width = 1100;
        Height = 800;
        MinimumSize = new Size(900, 680);
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 10);

        BuildUi();
        RefreshDevices();
        FormClosing += (_, _) => StopEverything();
    }

    private void BuildUi()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(18),
            ColumnCount = 1,
            RowCount = 7
        };
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        Controls.Add(root);

        root.Controls.Add(new Label
        {
            AutoSize = true,
            Text = "CommsAI",
            Font = new Font("Segoe UI", 22, FontStyle.Bold)
        });

        root.Controls.Add(new Label
        {
            AutoSize = true,
            MaximumSize = new Size(1000, 0),
            Text = "Local incoming voice translation for games. " +
                   "Select the Windows output carrying your game audio.",
            Margin = new Padding(0, 0, 0, 14)
        });

        var settings = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            ColumnCount = 4,
            Margin = new Padding(0, 0, 0, 12)
        };

        for (var i = 0; i < 4; i++)
            settings.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));

        settings.Controls.Add(LabelFor("Windows output"), 0, 0);
        settings.Controls.Add(LabelFor("Model"), 1, 0);
        settings.Controls.Add(LabelFor("GPU compute"), 2, 0);
        settings.Controls.Add(LabelFor("Playback"), 3, 0);

        _deviceBox.Dock = DockStyle.Fill;
        _deviceBox.DropDownStyle = ComboBoxStyle.DropDownList;

        _modelBox.Dock = DockStyle.Fill;
        _modelBox.DropDownStyle = ComboBoxStyle.DropDownList;
        _modelBox.Items.AddRange(
            ["large-v3", "distil-large-v3", "medium", "small"]
        );
        _modelBox.SelectedIndex = 0;

        _computeBox.Dock = DockStyle.Fill;
        _computeBox.DropDownStyle = ComboBoxStyle.DropDownList;
        _computeBox.Items.AddRange(["float16", "int8_float16"]);
        _computeBox.SelectedIndex = 0;

        _speakBox.AutoSize = true;
        _speakBox.Text = "Speak English";
        _speakBox.Checked = true;
        _speakBox.Margin = new Padding(4, 6, 0, 0);

        settings.Controls.Add(_deviceBox, 0, 1);
        settings.Controls.Add(_modelBox, 1, 1);
        settings.Controls.Add(_computeBox, 2, 1);
        settings.Controls.Add(_speakBox, 3, 1);
        root.Controls.Add(settings);

        var actionPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            Margin = new Padding(0, 0, 0, 12)
        };

        _refreshButton.Text = "Refresh devices";
        _refreshButton.AutoSize = true;
        _refreshButton.Click += (_, _) => RefreshDevices();

        _startButton.Text = "Start";
        _startButton.AutoSize = true;
        _startButton.Click += async (_, _) => await StartAsync();

        _stopButton.Text = "Stop";
        _stopButton.AutoSize = true;
        _stopButton.Enabled = false;
        _stopButton.Click += (_, _) => StopEverything();

        actionPanel.Controls.Add(_refreshButton);
        actionPanel.Controls.Add(_startButton);
        actionPanel.Controls.Add(_stopButton);

        _meter.Width = 280;
        _meter.Height = 26;
        _meter.Maximum = 1000;
        _meter.Margin = new Padding(20, 3, 0, 0);
        actionPanel.Controls.Add(_meter);

        _language.AutoSize = true;
        _language.Text = "Language: —";
        _language.Margin = new Padding(20, 7, 0, 0);
        actionPanel.Controls.Add(_language);

        _processing.AutoSize = true;
        _processing.Text = "Processing: —";
        _processing.Margin = new Padding(20, 7, 0, 0);
        actionPanel.Controls.Add(_processing);

        root.Controls.Add(actionPanel);

        var originalGroup = new GroupBox
        {
            Text = "Original-language transcript",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };
        _original.Dock = DockStyle.Fill;
        _original.Multiline = true;
        _original.ReadOnly = true;
        _original.ScrollBars = ScrollBars.Vertical;
        _original.Font = new Font("Segoe UI", 12);
        originalGroup.Controls.Add(_original);
        root.Controls.Add(originalGroup);

        var englishGroup = new GroupBox
        {
            Text = "Literal English translation",
            Dock = DockStyle.Fill,
            Padding = new Padding(10)
        };
        _english.Dock = DockStyle.Top;
        _english.Height = 130;
        _english.Multiline = true;
        _english.ReadOnly = true;
        _english.ScrollBars = ScrollBars.Vertical;
        _english.Font = new Font("Segoe UI", 12);
        englishGroup.Controls.Add(_english);

        _history.Dock = DockStyle.Fill;
        _history.ReadOnly = true;
        _history.AllowUserToAddRows = false;
        _history.AllowUserToDeleteRows = false;
        _history.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
        _history.ColumnHeadersHeightSizeMode =
            DataGridViewColumnHeadersHeightSizeMode.AutoSize;
        _history.Columns.Add("Time", "Time");
        _history.Columns.Add("Language", "Language");
        _history.Columns.Add("English", "English");
        _history.Columns[0].FillWeight = 15;
        _history.Columns[1].FillWeight = 15;
        _history.Columns[2].FillWeight = 70;
        _history.Top = 145;
        _history.Height = 180;
        _history.Anchor = AnchorStyles.Left | AnchorStyles.Right |
                          AnchorStyles.Top | AnchorStyles.Bottom;
        englishGroup.Controls.Add(_history);
        _history.BringToFront();

        root.Controls.Add(englishGroup);

        _status.Dock = DockStyle.Fill;
        _status.AutoSize = true;
        _status.BorderStyle = BorderStyle.Fixed3D;
        _status.Padding = new Padding(8);
        _status.Text = "Ready";
        root.Controls.Add(_status);
    }

    private static Label LabelFor(string text) => new()
    {
        AutoSize = true,
        Text = text,
        Margin = new Padding(0, 0, 0, 3)
    };

    private void RefreshDevices()
    {
        StopEverything();

        try
        {
            _deviceBox.Items.Clear();
            _enumerator?.Dispose();
            _enumerator = new MMDeviceEnumerator();

            var devices = _enumerator.EnumerateAudioEndPoints(
                DataFlow.Render,
                DeviceState.Active
            );

            foreach (var device in devices)
                _deviceBox.Items.Add(new DeviceItem(device));

            if (_deviceBox.Items.Count > 0)
            {
                var defaultDevice = _enumerator.GetDefaultAudioEndpoint(
                    DataFlow.Render,
                    Role.Multimedia
                );

                var index = 0;
                for (var i = 0; i < _deviceBox.Items.Count; i++)
                {
                    if (_deviceBox.Items[i] is DeviceItem item &&
                        item.Device.ID == defaultDevice.ID)
                    {
                        index = i;
                        break;
                    }
                }

                _deviceBox.SelectedIndex = index;
                SetStatus($"Found {_deviceBox.Items.Count} output device(s).");
            }
        }
        catch (Exception ex)
        {
            SetStatus($"Device error: {ex.Message}");
        }
    }

    private async Task StartAsync()
    {
        if (_deviceBox.SelectedItem is not DeviceItem selected)
        {
            MessageBox.Show("Select a Windows output device.");
            return;
        }

        try
        {
            SetControlsRunning(true);
            SetStatus("Starting local GPU backend…");

            _backend = new BackendClient();
            _backend.StatusReceived += message => Ui(() =>
            {
                SetStatus(message);
                if (message.Contains("ready", StringComparison.OrdinalIgnoreCase))
                {
                    _backendReady = true;
                    _ = ProcessNextSegmentAsync();
                }
            });
            _backend.ErrorReceived += message => Ui(() => SetStatus(message));
            _backend.ResultReceived += result => Ui(() => ShowResult(result));

            await _backend.StartAsync(
                _modelBox.Text,
                _computeBox.Text
            );

            var temp = Path.Combine(
                Environment.GetFolderPath(
                    Environment.SpecialFolder.LocalApplicationData
                ),
                "CommsAI",
                "segments"
            );
            Directory.CreateDirectory(temp);

            _capture = new WasapiLoopbackCapture(selected.Device);
            _segmenter = new AudioSegmenter(_capture.WaveFormat, temp);
            _segmenter.LevelMeasured += level => Ui(() =>
                _meter.Value = Math.Clamp((int)(level * 5000), 0, 1000)
            );
            _segmenter.SegmentReady += path =>
            {
                lock (_pendingSegments)
                    _pendingSegments.Enqueue(path);
                _ = ProcessNextSegmentAsync();
            };

            _capture.DataAvailable += (_, args) =>
                _segmenter.Push(args.Buffer, args.BytesRecorded);
            _capture.RecordingStopped += (_, args) =>
            {
                if (args.Exception is not null)
                    Ui(() => SetStatus($"Capture error: {args.Exception.Message}"));
            };

            _capture.StartRecording();
            SetStatus(
                "Loading the model and listening. First launch may take several minutes."
            );
        }
        catch (Exception ex)
        {
            SetStatus($"Start failed: {ex.Message}");
            StopEverything();
        }
    }

    private async Task ProcessNextSegmentAsync()
    {
        if (!_backendReady || _processingSegment || _backend is null)
            return;

        string? path = null;
        lock (_pendingSegments)
        {
            if (_pendingSegments.Count > 0)
                path = _pendingSegments.Dequeue();
        }

        if (path is null)
            return;

        _processingSegment = true;
        Ui(() => SetStatus("Transcribing captured speech…"));

        try
        {
            await _backend.TranscribeAsync(path);
        }
        catch (Exception ex)
        {
            Ui(() => SetStatus($"Transcription request failed: {ex.Message}"));
            TryDelete(path);
            _processingSegment = false;
            await ProcessNextSegmentAsync();
        }
    }

    private void ShowResult(BackendResult result)
    {
        _original.Text = result.Original;
        _english.Text = result.English;
        _language.Text =
            $"Language: {result.Language} ({result.LanguageProbability:P0})";
        _processing.Text =
            $"Processing: {result.ProcessingSeconds:0.00}s";

        _history.Rows.Insert(
            0,
            DateTime.Now.ToString("HH:mm:ss"),
            result.Language,
            result.English
        );

        if (_speakBox.Checked && !string.IsNullOrWhiteSpace(result.English))
        {
            _speech ??= new SpeechSynthesizer();
            _speech.SpeakAsyncCancelAll();
            _speech.SpeakAsync(result.English);
        }

        if (!string.IsNullOrWhiteSpace(result.SourcePath))
            TryDelete(result.SourcePath);

        _processingSegment = false;
        SetStatus("Listening…");
        _ = ProcessNextSegmentAsync();
    }

    private void StopEverything()
    {
        try
        {
            _capture?.StopRecording();
        }
        catch { }

        _capture?.Dispose();
        _capture = null;

        _segmenter?.Dispose();
        _segmenter = null;

        _backend?.Dispose();
        _backend = null;

        _speech?.SpeakAsyncCancelAll();
        _speech?.Dispose();
        _speech = null;

        _backendReady = false;
        _processingSegment = false;

        lock (_pendingSegments)
        {
            while (_pendingSegments.Count > 0)
                TryDelete(_pendingSegments.Dequeue());
        }

        _meter.Value = 0;
        SetControlsRunning(false);
    }

    private void SetControlsRunning(bool running)
    {
        _startButton.Enabled = !running;
        _stopButton.Enabled = running;
        _deviceBox.Enabled = !running;
        _modelBox.Enabled = !running;
        _computeBox.Enabled = !running;
        _refreshButton.Enabled = !running;
    }

    private void SetStatus(string text) => _status.Text = text;

    private void Ui(Action action)
    {
        if (IsDisposed)
            return;
        BeginInvoke(action);
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch { }
    }

    private sealed class DeviceItem
    {
        public DeviceItem(MMDevice device) => Device = device;
        public MMDevice Device { get; }
        public override string ToString() => Device.FriendlyName;
    }
}
