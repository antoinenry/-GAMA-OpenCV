using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Emgu.CV;
using Emgu.CV.CvEnum;

public class NewBehaviourScript : MonoBehaviour
{
    public bool grayscaleEnabled;

    [Header("Adaptive Threshold")]
    public double threshHoldMax;
    public AdaptiveThresholdType adaptiveType;
    public ThresholdType thresholdType;
    [Range(1, 1000)]
    public int blockSize;
        public double param1;

    private VideoCapture vCapture;

    private void Start()
    {
        vCapture = new VideoCapture();
    }
    void Update()
    {
        Mat mCapture = vCapture.QueryFrame();

        // NDG
        if (grayscaleEnabled)
            CvInvoke.CvtColor(mCapture, mCapture, ColorConversion.Bgr2Gray);

        // THRSH
        CvInvoke.AdaptiveThreshold(mCapture, mCapture, threshHoldMax, adaptiveType, thresholdType, blockSize*2+1, param1);

        CvInvoke.Imshow("Display truc", mCapture);
    }
}
