import tesseract

def recognize(mImgFile):
    api = tesseract.TessBaseAPI()
    api.Init(".","eng",tesseract.OEM_DEFAULT)
    eng = "0123456789abcdefghijklmnopqrstuvwxyz."
    digit = "0123456789"
    api.SetVariable("tessedit_char_whitelist", digit)
    api.SetPageSegMode(tesseract.PSM_AUTO)

    mBuffer=open(mImgFile,"rb").read()
    result = tesseract.ProcessPagesBuffer(mBuffer,len(mBuffer),api)
    print "result(ProcessPagesBuffer)=",result
    api.End()

if __name__ == '__main__':
    recognize(mImgFile = "train/6.jpg")
