using System.Diagnostics;
using System.Text.Json;

namespace CommsAI;

public sealed class BackendClient : IDisposable
{
    private Process? _process;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly CancellationTokenSource _cts = new();

    public event Action<BackendResult>? ResultReceived;
    public event Action<string>? StatusReceived;
    public event Action<string>? ErrorReceived;

    public bool IsRunning => _process is { HasExited: false };

    public async Task StartAsync(string model, string computeType)
    {
        if (IsRunning)
            return;

        var backendPath = Path.Combine(
            AppContext.BaseDirectory,
            "backend",
            "CommsAI.Backend.exe"
        );

        if (!File.Exists(backendPath))
            throw new FileNotFoundException(
                "The packaged AI backend was not found.",
                backendPath
            );

        var startInfo = new ProcessStartInfo
        {
            FileName = backendPath,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = AppContext.BaseDirectory
        };

        _process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        _process.Exited += (_, _) =>
            ErrorReceived?.Invoke("The AI backend stopped unexpectedly.");

        if (!_process.Start())
            throw new InvalidOperationException("Could not start the AI backend.");

        _ = Task.Run(() => ReadOutputLoopAsync(_process, _cts.Token));
        _ = Task.Run(() => ReadErrorLoopAsync(_process, _cts.Token));

        await SendAsync(new
        {
            command = "load",
            model,
            compute_type = computeType
        });
    }

    public Task TranscribeAsync(string wavPath) =>
        SendAsync(new
        {
            command = "transcribe",
            path = wavPath
        });

    private async Task SendAsync(object payload)
    {
        if (!IsRunning || _process is null)
            throw new InvalidOperationException("The AI backend is not running.");

        var json = JsonSerializer.Serialize(payload);

        await _sendLock.WaitAsync();
        try
        {
            await _process.StandardInput.WriteLineAsync(json);
            await _process.StandardInput.FlushAsync();
        }
        finally
        {
            _sendLock.Release();
        }
    }

    private async Task ReadOutputLoopAsync(Process process, CancellationToken token)
    {
        while (!token.IsCancellationRequested && !process.HasExited)
        {
            var line = await process.StandardOutput.ReadLineAsync(token);
            if (line is null)
                break;

            try
            {
                var envelope = JsonSerializer.Deserialize<BackendEnvelope>(
                    line,
                    JsonOptions
                );

                if (envelope?.Type == "result" && envelope.Result is not null)
                    ResultReceived?.Invoke(envelope.Result);
                else if (envelope?.Message is not null)
                    StatusReceived?.Invoke(envelope.Message);
            }
            catch (Exception ex)
            {
                ErrorReceived?.Invoke($"Backend response error: {ex.Message}");
            }
        }
    }

    private async Task ReadErrorLoopAsync(Process process, CancellationToken token)
    {
        while (!token.IsCancellationRequested && !process.HasExited)
        {
            var line = await process.StandardError.ReadLineAsync(token);
            if (line is null)
                break;

            ErrorReceived?.Invoke(line);
        }
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public void Dispose()
    {
        _cts.Cancel();

        try
        {
            if (_process is { HasExited: false })
                _process.Kill(entireProcessTree: true);
        }
        catch
        {
            // Best-effort shutdown.
        }

        _process?.Dispose();
        _sendLock.Dispose();
        _cts.Dispose();
    }
}

public sealed class BackendEnvelope
{
    public string? Type { get; set; }
    public string? Message { get; set; }
    public BackendResult? Result { get; set; }
}

public sealed class BackendResult
{
    public string Language { get; set; } = "unknown";
    public double LanguageProbability { get; set; }
    public string Original { get; set; } = "";
    public string English { get; set; } = "";
    public double ProcessingSeconds { get; set; }
    public string? SourcePath { get; set; }
}
