function scrollToUpload(){
    document.getElementById("upload-section")
    .scrollIntoView({behavior:"smooth"})
}

// FAQ
document.querySelectorAll(".faq-question").forEach(button => {
    button.addEventListener("click", () => {
        const answer = button.nextElementSibling
        answer.style.display =
        answer.style.display === "block" ? "none" : "block"
    })
})

// File name
const uploadInput = document.getElementById("audio-upload")
const fileName = document.getElementById("file-name")

if(uploadInput){
    uploadInput.addEventListener("change", () => {
        if(uploadInput.files.length > 0){
            fileName.textContent = uploadInput.files[0].name
        }
    })
}

// PROGRESS + FETCH SUBMIT
const form = document.getElementById("uploadForm")

if(form){
    form.addEventListener("submit", async function(e){

        e.preventDefault()   // STOP normal form submit

        const overlay = document.getElementById("loading-overlay")
        const fill = document.getElementById("progress-fill")
        const text = document.getElementById("progress-text")

        overlay.style.display = "flex"

        let progress = 0

        const interval = setInterval(() => {
            progress += 5
            fill.style.width = progress + "%"

            if(progress < 30) text.innerText = "Uploading Audio..."
            else if(progress < 60) text.innerText = "Transcribing..."
            else if(progress < 85) text.innerText = "Generating Notes..."
            else text.innerText = "Finalizing PDF..."

            if(progress >= 95){
                clearInterval(interval)
            }
        }, 500)

        try {
            const formData = new FormData(form)

            const response = await fetch("/process", {
                method: "POST",
                body: formData
            })

            const blob = await response.blob()

            const url = window.URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            a.download = "Lecture_Notes.pdf"
            document.body.appendChild(a)
            a.click()
            a.remove()

            window.URL.revokeObjectURL(url)

        } catch (error) {
            alert("Error generating notes.")
        }

        // Finish progress
        fill.style.width = "100%"
        text.innerText = "Completed!"

        setTimeout(() => {
            overlay.style.display = "none"
            fill.style.width = "0%"
        }, 1000)

    })
}