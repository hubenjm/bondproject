from PIL import Image
import tesserocr

imagefile = 'cusipimage.png'
f = Image.open(imagefile)
old_size = f.size
pad = 7
new_size = (old_size[0] + 2*pad, old_size[1] + 2*pad)
new_im = Image.new("RGB", new_size)
new_im.paste(f, ((new_size[0] - old_size[0])/2, (new_size[1] - old_size[1])/2))

pixels = new_im.load()

for i in range(7):
    for j in range(pad, old_size[1]+pad): #left and right bands
        pixels[i,j] = (255,255,255)
        pixels[new_size[1]-1-i,j] = (255,255,255)

for i in xrange(104):
    for j in xrange(7): #top and bottom bands
        pixels[i,j] = (255,255,255)
        pixels[i,new_size[0]-1-j] = (255,255,255)

s = tesserocr.image_to_text(new_im)[:9]
print s
