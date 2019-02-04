using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

using UnityEngine;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Structure;

public class NewBehaviourScript : MonoBehaviour
{
    public enum MORPHO_OPERATION { NONE, ERODE, DILATE, OPEN, CLOSE };
    public enum STEP { RAW, GRAY, THRESH, MORPHO, CONTOUR };

    public STEP displayResult;

    [Header("Adaptive Threshold")]
    public double threshHoldMax;
    public AdaptiveThresholdType adaptiveType;
    public ThresholdType thresholdType;
    [Range(1, 1000)] public int blockSize;
    public double param1;

    [Header("Morphology operations")]
    public MORPHO_OPERATION morphOperation;
    public ElementShape structuringShape;
    [Range(1, 10)] public int structuringSize;
    [Range(1, 10)] public int iterationsCount;

    [Header("Contour detection")]
    public ChainApproxMethod contourChainApproxMethod;
    public bool showContours;
    
    private VideoCapture vCapture;

    private void Start()
    {
        vCapture = new VideoCapture();
    }
    void Update()
    {
        Mat mCapture = vCapture.QueryFrame();
        Mat mDisplay;
        mDisplay = mCapture.Clone();

        // NDG
        CvInvoke.CvtColor(mCapture, mCapture, ColorConversion.Bgr2Gray);
        if (displayResult == STEP.GRAY) mDisplay = mCapture.Clone();

        // THRSH
        CvInvoke.AdaptiveThreshold(mCapture, mCapture, threshHoldMax, adaptiveType, thresholdType, blockSize*2+1, param1);
        if (displayResult == STEP.THRESH) mDisplay = mCapture.Clone();

        // MORPHO
        Mat structuringElement = CvInvoke.GetStructuringElement(structuringShape, new Size(structuringSize * 2 + 1, structuringSize * 2 + 1), new Point(-1, -1));
        switch (morphOperation)
        {
            case MORPHO_OPERATION.DILATE:
                CvInvoke.Dilate(mCapture, mCapture, structuringElement, new Point(-1, -1), iterationsCount, BorderType.Constant, new MCvScalar(0));
                break;

            case MORPHO_OPERATION.ERODE:
                CvInvoke.Erode(mCapture, mCapture, structuringElement, new Point(-1, -1), iterationsCount, BorderType.Constant, new MCvScalar(0));
                break;

            case MORPHO_OPERATION.OPEN:
                CvInvoke.Dilate(mCapture, mCapture, structuringElement, new Point(-1, -1), iterationsCount, BorderType.Constant, new MCvScalar(0));
                CvInvoke.Erode(mCapture, mCapture, structuringElement, new Point(-1, -1), iterationsCount, BorderType.Constant, new MCvScalar(0));
                break;

            case MORPHO_OPERATION.CLOSE:
                CvInvoke.Erode(mCapture, mCapture, structuringElement, new Point(-1, -1), iterationsCount, BorderType.Constant, new MCvScalar(0));
                CvInvoke.Dilate(mCapture, mCapture, structuringElement, new Point(-1, -1), iterationsCount, BorderType.Constant, new MCvScalar(0));
                break;
        }
        if (displayResult == STEP.MORPHO) mDisplay = mCapture.Clone();

        CvInvoke.CvtColor(mDisplay, mDisplay, ColorConversion.Gray2Rgb
             );

        // CONTOUR
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        Mat hierarchy = new Mat();
        CvInvoke.FindContours(mCapture, contours, hierarchy, RetrType.List, contourChainApproxMethod);
        if (showContours)
        {
            for (int i = 0; i < contours.Size; i++)
                CvInvoke.DrawContours(mDisplay, contours, i, new MCvScalar(0, 255, 255), 2);
        }

        // DISPLAY
        CvInvoke.Imshow("Display truc", mDisplay);
    }

    private void OnDestroy()
    {
        vCapture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}
