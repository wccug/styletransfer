using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Diagnostics;
using System.Reflection.Emit;

namespace styletransfer
{
    internal class Program
    {
        // Models that come from ONNX Model Zoo
        static string[] styleNames = { "mosaic", "candy", "udnie", "pointilism" };

        // If the resolution of the model is kept pretty small it can be done on CPU even on a
        // not super high end computer.
        const int ModelWidth = 224;
        const int ModelHeight = 224;

        struct NamedInferenceSession
        {
            public string Name;
            public InferenceSession InferenceSession;
            public NamedInferenceSession(string n, InferenceSession i) { Name = n; InferenceSession = i; }
        }
        static List<NamedInferenceSession> LoadModels()
        {
            List<NamedInferenceSession> models = new List<NamedInferenceSession>();
            for (int i = 0; i < styleNames.Length; ++i)
            {
                string styleName = styleNames[i];
                string path = Path.Combine(AppContext.BaseDirectory, "models", $"{styleName}-9.onnx");

                if (File.Exists(path))
                {
                    models.Add(new NamedInferenceSession(styleNames[i], new InferenceSession(path)));
                }
                else
                {
                    Console.WriteLine($"Missing model: {path}");
                }
            }
            return models;
        }

        static Mat Stylize(NamedInferenceSession session, Mat cameraInputBgrUnorm)
        {
            // Scale the camera input down to fit the model resolution
            using Mat scaledCameraInputBgrUnorm = new Mat();
            Cv2.Resize(cameraInputBgrUnorm, scaledCameraInputBgrUnorm, new Size(ModelWidth, ModelHeight));

            // OpenCV's camera format is BGR, but the style models were trained on RGB images,
            // so convert BGR to RGB.
            using Mat scaledCameraInputRgbUnorm = new Mat();
            Cv2.CvtColor(scaledCameraInputBgrUnorm, scaledCameraInputRgbUnorm, ColorConversionCodes.BGR2RGB);

            // OpenCV's camera format colors are 8bits per channel, but the style models are float32 per channel.
            // The models do operate on [0-255] range though, so we do not need to scale the values.
            using Mat scaledCameraInputRgbF32 = new Mat();
            scaledCameraInputRgbUnorm.ConvertTo(scaledCameraInputRgbF32, MatType.CV_32FC3);
            Mat[] scaledCameraInputRgbF32Planes = Cv2.Split(scaledCameraInputRgbF32);

            DenseTensor<float> tensorInput = new DenseTensor<float>(new[] { 1, 3, ModelWidth, ModelHeight });

            for (int channel = 0; channel < 3; ++channel)
            {
                // Turn each channel into linear tensor data so it's easier to operate on
                scaledCameraInputRgbF32Planes[channel].GetArray(out float[] data);
                data.CopyTo(tensorInput.Buffer.Span.Slice(channel * ModelWidth * ModelHeight));
                scaledCameraInputRgbF32Planes[channel].Dispose();
            }

            string inputName = session.InferenceSession.InputMetadata.Keys.First();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> inferenceResult = session.InferenceSession.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensorInput) });
            Tensor<float> resultTensor = inferenceResult.First().AsTensor<float>();
            float[] resultData = resultTensor.ToArray();

            // Merge the planes back together
            var resultPlanes = new Mat[3];
            for (int channel = 0; channel < 3; ++channel)
            {
                var plane = new float[ModelWidth * ModelHeight];
                Array.Copy(resultData, channel * ModelWidth * ModelHeight, plane, 0, plane.Length);
                resultPlanes[channel] = Mat.FromPixelData(ModelWidth, ModelHeight, MatType.CV_32FC1, plane).Clone();
            }
            using var mergedPlanes = new Mat();
            Cv2.Merge(resultPlanes, mergedPlanes);
            foreach (var p in resultPlanes) p.Dispose();

            // The model doesn't guarantee the output stays within [0-255], so clamp anything below 0 to 0 and anything above 255 to 255.
            using var resultUnorm = new Mat();
            mergedPlanes.ConvertTo(resultUnorm, MatType.CV_8UC3);

            // Convert the result back to BGRUnorm so that it can be displayed by OpenCV
            Mat bgrResult = new Mat();
            Cv2.CvtColor(resultUnorm, bgrResult, ColorConversionCodes.RGB2BGR);
            return bgrResult;
        }

        static void Main(string[] args)
        {
            List<NamedInferenceSession> sessions = LoadModels();
            if (sessions.Count == 0)
            {
                Console.WriteLine("No models found. Put the .onnx files in a 'models' folder.");
                return;
            }

            using VideoCapture videoCapture = new VideoCapture(0); // Use the default camera
            if (!videoCapture.IsOpened())
            {
                Console.WriteLine("Could not open webcam.");
                return;
            }

            const string windowName = "Hackforge Style Transfer Demo";
            Cv2.NamedWindow(windowName, WindowFlags.Normal | WindowFlags.KeepRatio);
            Cv2.SetWindowProperty(windowName, WindowPropertyFlags.Fullscreen, (double)WindowFlags.FullScreen);

            Stopwatch stopwatch = new Stopwatch(); // Used for FPS counter
            Mat frame = new Mat(); // OpenCV matrix type
            int currentModel = 0; // -1 means show the raw camera feed
            bool errorReadingCamera = false;
            float fps = 0;
            bool fullscreen = true;
            bool showDiagnostics = true;

            while (true)
            {
                stopwatch.Restart();

                if (!videoCapture.Read(frame) || frame.Empty())
                {
                    errorReadingCamera = true;
                    break;
                }

                Cv2.Flip(frame, frame, FlipMode.Y); // Mirror

                Mat rendered;
                if (currentModel == -1)
                {
                    rendered = frame.Clone();
                }
                else
                {
                    rendered = Stylize(sessions[currentModel], frame);
                }

                // Scale up to the full display size, but keep aspect ratio
                Rect r = Cv2.GetWindowImageRect(windowName);
                int windowWidth = r.Width;
                int windowHeight = r.Height;

                float aspect = (float)frame.Width / frame.Height;
                int displayWidth = windowWidth;
                int displayHeight = (int)(windowWidth / aspect);
                if (displayHeight > windowHeight)
                {
                    displayHeight = windowHeight;
                    displayWidth = (int)(displayHeight * aspect);
                }

                using var display = new Mat();
                Cv2.Resize(rendered, display, new Size(displayWidth, displayHeight), interpolation: InterpolationFlags.Cubic);
                rendered.Dispose();

                if (showDiagnostics)
                {
                    string label = currentModel > 0 ? sessions[currentModel].Name : "raw";
                    Cv2.PutText(display, $"{label}  {fps:F1} FPS", new Point(10, 30), HersheyFonts.HersheySimplex, 0.8, Scalar.White, 2);
                }

                Cv2.ImShow(windowName, display);

                int key = Cv2.WaitKey(1);
                if (key == 'q' || key == (int)ConsoleKey.Escape)
                {
                    break;
                }
                if (key == '0')
                {
                    currentModel = -1;
                }
                else if (key >= '1' && key < '1' + sessions.Count)
                {
                    currentModel = key - '1';
                }
                else if (key == 'f' || key == 'F')
                {
                    // Toggle fullscreen/windowed mode
                    fullscreen = !fullscreen;
                    if (fullscreen)
                    {
                        Cv2.SetWindowProperty(windowName, WindowPropertyFlags.Fullscreen, (double)WindowFlags.FullScreen);
                    }
                    else
                    {
                        Cv2.SetWindowProperty(windowName, WindowPropertyFlags.Fullscreen, (double)WindowFlags.Normal);
                    }
                }
                else if (key == 'd' || key == 'd')
                {
                    showDiagnostics = !showDiagnostics;
                }

                // Exit when the user closes the window with the X button
                if (Cv2.GetWindowProperty(windowName, WindowPropertyFlags.Visible) < 1)
                {
                    break;
                }

                fps = 1000.0f / Math.Max(1, stopwatch.ElapsedMilliseconds);

            }

            if (errorReadingCamera)
            {
                Console.WriteLine("An error occurred while reading camera data.");
                return;
            }

            Console.WriteLine("Finished successfully.");
        }
    }
}
